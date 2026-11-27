"""实验2：基于 TextCNN 的情感分类（A/B 对照）

做什么
------
同一份 Amazon 评论、同一个 TextCNN、同一套超参，只改词嵌入的起点：
    A 组：词嵌入随机初始化
    B 组：词嵌入用实验1 window=5 的词向量填充
然后比较两组在测试集上的 Accuracy 与 Macro-F1。

用法
----
    python main.py train --train-csv ... --dev-csv ... --test-csv ... --handoff ... --output-dir ...
    python main.py predict --output-dir ... "一句英文评论"

"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# 复用实验1 的分词与 CSV 读取，保证两个实验的预处理完全一致
LAB1 = Path(__file__).resolve().parents[1] / "exp01-word2vec"
sys.path.insert(0, str(LAB1))
from lab_common import (PREPROCESSING, read_reviews, require_training_file,
                        sha256, tokenize)

# ===== 固定超参：与老师参考实现保持一致 =====
MAX_LENGTH = 100          # 每条评论统一 100 个词位，短补零，长截断
BATCH_SIZE = 64
KERNELS = (3, 4, 5)       # 三种卷积窗口宽度
FILTERS = 100             # 每种宽度用 100 个滤波器
DROPOUT = 0.5
LR = 0.001
LABELS = ("negative", "positive")
DEFAULT_SEED = 42
DEFAULT_EPOCHS = 5


class TextCNN(nn.Module):
    """三层卷积 + 掩码池化 + 全连接。"""

    def __init__(self, vocab_size, embedding_size):
        super().__init__()
        # padding_idx=0：第 0 行（PAD）的向量永远全零，且不参与梯度更新
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        # 每种窗口宽度一个卷积：输入 1 个通道，输出 100 个特征图
        self.convolutions = nn.ModuleList(
            nn.Conv2d(1, FILTERS, (width, embedding_size)) for width in KERNELS
        )
        self.classifier = nn.Sequential(nn.Dropout(DROPOUT),
                                        nn.Linear(FILTERS * len(KERNELS), 2))

    def forward(self, token_ids):
        embedded = self.embedding(token_ids).unsqueeze(1)   # 变成单通道"图"
        lengths = token_ids.ne(0).sum(dim=1)                # 每条评论的真实词数

        features = []
        for width, conv in zip(KERNELS, self.convolutions):
            activated = conv(embedded).squeeze(-1).relu()
            # 补零出来的位置不能参与取最大值，否则短句会被 0 污染
            valid_starts = (lengths - width + 1).clamp_min(1)
            valid = torch.arange(activated.size(-1),
                                 device=token_ids.device) < valid_starts[:, None]
            pooled = activated.masked_fill(~valid[:, None, :], 0).amax(dim=-1)
            features.append(pooled)

        return self.classifier(torch.cat(features, dim=1))


def load_handoff(folder, train_file):
    """读取实验1 导出的词向量，并核验它确实来自这批训练数据。"""
    vocab = json.loads((folder / "vocab.json").read_text(encoding="utf-8"))
    preprocessing = json.loads((folder / "preprocessing.json").read_text(encoding="utf-8"))
    manifest = json.loads((folder.parent / "run_manifest.json").read_text(encoding="utf-8"))

    # 三个校验：预处理规则一致、训练数据指纹一致、词表结构正确
    if preprocessing != PREPROCESSING or manifest["training_sha256"] != sha256(train_file):
        raise ValueError("词向量与训练数据不匹配，请用实验1 同一次训练的产物。")
    tokens = vocab["id_to_token"]
    ids = vocab["token_to_id"]
    if (vocab["pad_id"], vocab["unk_id"]) != (0, 1) or ids != {t: i for i, t in enumerate(tokens)}:
        raise ValueError("词表映射不符合约定：PAD 必须是 0，UNK 必须是 1。")

    matrix = np.load(folder / "embedding.npy", allow_pickle=False)
    if matrix.shape != (len(tokens), vocab["vector_size"]):
        raise ValueError("词向量矩阵的形状与词表长度不一致。")
    if not np.isfinite(matrix).all():
        raise ValueError("词向量里出现了 NaN 或 Inf。")
    if matrix[0].any():
        raise ValueError("PAD 行必须全零。")
    return vocab, torch.tensor(matrix, dtype=torch.float32)


def load_splits(train_file, dev_file, test_file):
    """读三份数据，并做泄漏检查。"""
    require_training_file(train_file)
    train, dev, test = (list(read_reviews(p)) for p in (train_file, dev_file, test_file))

    keys = lambda rows: {row.key() for row in rows}
    test_keys = keys(test)
    if keys(train) & (keys(dev) | test_keys):
        raise ValueError("训练集与开发集/测试集有重叠，实验1 的 --exclude-csv 没生效。")

    # 原始 dev 和 test 之间可能有一条同文评论，保留 test 不动，从 dev 剔除
    clean_dev = [row for row in dev if row.key() not in test_keys]

    for name, rows in (("train", train), ("dev", clean_dev), ("test", test)):
        if {row.label for row in rows} != {1, 2}:
            raise ValueError(f"{name} 必须同时包含负面(1) 和正面(2) 两类。")
    return train, clean_dev, test, len(dev) - len(clean_dev)


def encode(tokens, vocab):
    """把一句评论转成固定长度 100 的词编号，词表外的词用 UNK(1)。"""
    ids = [vocab["token_to_id"].get(t, 1) for t in tokens[:MAX_LENGTH]]
    return ids + [0] * (MAX_LENGTH - len(ids))


def to_dataset(rows, vocab):
    ids = torch.tensor([encode(row.tokens(), vocab) for row in rows], dtype=torch.long)
    labels = torch.tensor([row.label - 1 for row in rows], dtype=torch.long)  # 1/2 → 0/1
    return TensorDataset(ids, labels)


def compute_metrics(target, prediction):
    """混淆矩阵的行是真实类别，列是预测类别，顺序为负面、正面。"""
    matrix = torch.bincount(target * 2 + prediction, minlength=4).reshape(2, 2)
    tp = matrix.diag()
    denom = (matrix.sum(0) + matrix.sum(1)).clamp_min(1)   # 2TP+FP+FN
    f1 = 2 * tp / denom
    return {"accuracy": float(tp.sum() / matrix.sum()),
            "macro_f1": float(f1.mean()),
            "confusion_matrix": matrix.tolist()}


@torch.inference_mode()
def evaluate(model, loader, loss_fn):
    model.eval()
    total, preds, targets = 0.0, [], []
    for x, y in loader:
        logits = model(x)
        total += loss_fn(logits, y).item() * y.size(0)
        preds.append(logits.argmax(dim=1))
        targets.append(y)
    target, prediction = torch.cat(targets), torch.cat(preds)
    return {**compute_metrics(target, prediction), "loss": total / len(target)}, prediction.tolist()


def train_arm(name, splits, initial, vocab, epochs, seed):
    """训练一组模型。

    initial 为 None 时是 A 组（随机初始化）；
    传入实验1 的词向量矩阵时是 B 组。
    这是 A/B 唯一的差别。
    """
    train, dev, test = splits

    # 每组都重置种子，保证两组的权重初始化、dropout、打乱顺序完全一致
    torch.manual_seed(seed)
    model = TextCNN(len(vocab["id_to_token"]), vocab["vector_size"])
    if initial is not None:
        with torch.no_grad():
            # 只覆盖第 2 行往后：第 0 行 PAD 保持全零，第 1 行 UNK 两组共用同一个随机初值
            model.embedding.weight[2:].copy_(initial[2:])

    train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True,
                              generator=torch.Generator().manual_seed(seed))
    dev_loader = DataLoader(dev, batch_size=BATCH_SIZE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_f1, best_epoch, best_state, history = -1.0, 1, None, []
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        for x, y in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * y.size(0)

        dev_score, _ = evaluate(model, dev_loader, loss_fn)
        history.append({"model": name, "epoch": epoch,
                        "train_loss": loss_sum / len(train),
                        "dev_loss": dev_score["loss"],
                        "dev_macro_f1": dev_score["macro_f1"]})
        print(f"{name} epoch {epoch}: dev macro-F1 {dev_score['macro_f1']:.4f}", flush=True)

        if dev_score["macro_f1"] > best_f1:
            best_f1 = dev_score["macro_f1"]
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    # 用开发集选出的最佳轮次，在测试集上只评一次
    model.load_state_dict(best_state)
    test_score, predictions = evaluate(model, DataLoader(test, batch_size=BATCH_SIZE), loss_fn)
    return model, {"best_dev_epoch": best_epoch, **test_score}, predictions, history


def plot_curves(history, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for name in ("A_random", "B_word2vec"):
        rows = [r for r in history if r["model"] == name]
        for ax, key, ylabel in zip(axes, ("train_loss", "dev_macro_f1"),
                                   ("Train loss", "Dev Macro-F1")):
            ax.plot([r["epoch"] for r in rows], [r[key] for r in rows],
                    marker="o", label=name)
            ax.set(xlabel="Epoch", ylabel=ylabel)
            ax.legend()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def train(args):
    train_rows, dev_rows, test_rows, removed = load_splits(
        args.train_csv, args.dev_csv, args.test_csv)
    vocab, embedding = load_handoff(args.handoff, args.train_csv)
    splits = tuple(to_dataset(rows, vocab) for rows in (train_rows, dev_rows, test_rows))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)   # 必须是新目录，避免覆盖旧结果

    print(f"数据：train {len(train_rows)} / dev {len(dev_rows)} / test {len(test_rows)}"
          f"（从 dev 剔除与 test 重复的 {removed} 条）")
    print(f"训练：epochs={args.epochs}, seed={args.seed}, 词表 {len(vocab['id_to_token'])} 词\n")

    report = {
        "counts": {"train": len(train_rows), "dev": len(dev_rows),
                   "test": len(test_rows), "dev_removed_test_overlap": removed},
        "label_order": list(LABELS), "seed": args.seed, "epochs": args.epochs,
        "input_sha256": {n: sha256(f) for n, f in (
            ("train", args.train_csv), ("dev", args.dev_csv),
            ("test", args.test_csv), ("embedding", args.handoff / "embedding.npy"))},
        "models": {},
    }

    history, predictions = [], {}
    for name, initial in (("A_random", None), ("B_word2vec", embedding)):
        model, report["models"][name], predictions[name], series = train_arm(
            name, splits, initial, vocab, args.epochs, args.seed)
        torch.save(model.state_dict(), output_dir / f"{name}.pt")
        history.extend(series)

    # 只记录至少一组判错的评论，供错误分析使用
    errors = []
    for review, a, b in zip(test_rows, predictions["A_random"], predictions["B_word2vec"]):
        gold = review.label - 1
        if a == b == gold:
            continue
        tokens = review.tokens()[:MAX_LENGTH]
        errors.append({"gold": LABELS[gold], "A": LABELS[a], "B": LABELS[b],
                       "title": review.title, "text": review.text,
                       "oov_count": sum(t not in vocab["token_to_id"] for t in tokens)})

    (output_dir / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "vocab.json").write_text(
        json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8")

    with open(output_dir / "history.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "epoch", "train_loss",
                                               "dev_loss", "dev_macro_f1"])
        writer.writeheader()
        writer.writerows(history)

    with open(output_dir / "errors.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["gold", "A", "B", "title", "text", "oov_count"])
        writer.writeheader()
        writer.writerows(errors)

    plot_curves(history, output_dir / "curves.png")

    print("\n" + "=" * 56)
    print(f"{'指标':<12}{'A_random':>14}{'B_word2vec':>14}{'差值':>10}")
    for key, label in (("accuracy", "Accuracy"), ("macro_f1", "Macro-F1")):
        a = report["models"]["A_random"][key]
        b = report["models"]["B_word2vec"][key]
        print(f"{label:<12}{a:>14.4f}{b:>14.4f}{b - a:>+10.4f}")
    print(f"{'最佳轮次':<12}{report['models']['A_random']['best_dev_epoch']:>14}"
          f"{report['models']['B_word2vec']['best_dev_epoch']:>14}")
    print("=" * 56)
    print(f"结果已写入 {output_dir}")


@torch.inference_mode()
def predict(args):
    """现场比较：同一句话分别喂给 A 和 B。"""
    vocab = json.loads((Path(args.output_dir) / "vocab.json").read_text(encoding="utf-8"))
    tokens = tokenize(args.text)
    if not tokens:
        raise ValueError("请输入至少包含一个英文词的评论。")
    x = torch.tensor([encode(tokens, vocab)], dtype=torch.long)

    print(f"输入：{args.text}")
    print(f"分词后 {len(tokens)} 个词，截断到 {MAX_LENGTH} 个词位\n")
    for name in ("A_random", "B_word2vec"):
        model = TextCNN(len(vocab["id_to_token"]), vocab["vector_size"])
        model.load_state_dict(torch.load(Path(args.output_dir) / f"{name}.pt",
                                         map_location="cpu", weights_only=True))
        probs = model.eval()(x).softmax(dim=-1)[0]
        print(name, LABELS[int(probs.argmax())],
              {label: round(float(p), 4) for label, p in zip(LABELS, probs)})


def main():
    parser = argparse.ArgumentParser(description="实验2 TextCNN 情感分类 A/B 对照")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train")
    for name in ("train-csv", "dev-csv", "test-csv", "handoff", "output-dir"):
        p_train.add_argument(f"--{name}", required=True, type=Path)
    p_train.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p_train.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)

    p_pred = sub.add_parser("predict")
    p_pred.add_argument("--output-dir", required=True, type=Path)
    p_pred.add_argument("text")

    args = parser.parse_args()
    torch.set_num_threads(2)   # 限制线程数，让结果更可复现
    if args.command == "train":
        if args.epochs < 1:
            parser.error("--epochs 至少为 1")
        train(args)
    else:
        predict(args)


if __name__ == "__main__":
    main()
