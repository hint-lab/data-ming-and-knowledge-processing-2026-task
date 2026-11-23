"""实验 1 · 词向量表示与相似度分析 —— 核心代码

学号：25124294    姓名：吴欣阳
课程：2026 数据挖掘与知识处理

本脚本完成实验的完整流程：
    1. 读取并清洗 Amazon 评论语料：NFKC 归一化、去 HTML 标签、转小写、正则分词；
    2. 统计词频，用 min_count 截断长尾低频词，得到训练词表；
    3. 固定其余全部超参数，只改变滑动窗口 window=2 / window=5，训练两个 CBOW 模型；
    4. 实测 5 组指定词对的余弦相似度，输出指定词的 Top-5 语义近邻；
    5. 导出 window=5 的词向量矩阵与词表映射，供实验 2 直接加载。

运行方式：
    python main.py --csv data/train_sample.csv --out outputs
    python main.py --demo            # 没有语料时，用内置小语料跑通完整流程

依赖：Python 3.9+，gensim，numpy
数据：Amazon 评论训练集体积较大，未随作业提交；没有数据时可用 --demo 验证流程。
说明：gensim 的初始化与 Python 的字符串哈希有关。若要逐位复现实验报告中的数值，
      先设置环境变量 PYTHONHASHSEED=42 再运行（Windows: set PYTHONHASHSEED=42）。
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import os
import re
import subprocess
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# 实验约定：这些值在两组模型之间必须完全一致，唯一变量只有 window
# ---------------------------------------------------------------------------
WINDOWS = (2, 5)
PAIRS = [
    ("good", "great"),
    ("good", "bad"),
    ("phone", "battery"),
    ("expensive", "cheap"),
    ("screen", "battery"),
]
QUERIES = ["good", "bad", "great", "phone", "battery", "screen", "price", "quality", "easy", "poor"]

PREPROCESSING = {
    "version": 1,
    "language": "English",
    "unicode": "NFKC",
    "lowercase": True,
    "html_unescape": True,
    "remove_html_tags": True,
    "token_pattern": r"[a-z]+(?:'[a-z]+)?|[0-9]+",
    "remove_stopwords": False,
    "combine": "title + space + text",
}

# CBOW、维度 128、10 轮、负采样 5、下采样 1e-3、单线程（保证可复现）
FIXED_PARAMETERS = {
    "sg": 0,
    "workers": 1,
    "hs": 0,
    "negative": 5,
    "sample": 1e-3,
    "alpha": 0.025,
    "min_alpha": 1e-4,
}

# 没有数据集时用于演示流程的小语料（每个词都出现多次，便于看到近邻）
DEMO_SENTENCES = [
    "the battery life is great and the screen is bright .",
    "the battery life is bad and the screen is dim .",
    "this phone has a great screen and a good battery .",
    "this phone has a bad screen and a poor battery .",
    "the camera is great and the price is fair .",
    "the camera is bad and the price is high .",
    "i love this phone because the battery is great .",
    "i hate this phone because the battery is bad .",
    "the screen is bright and the battery is good .",
    "the screen is dim and the battery is poor .",
    "great battery , great screen , great phone .",
    "bad battery , bad screen , bad phone .",
    "the charger is good and the cable is great .",
    "the charger is bad and the cable is poor .",
    "this laptop has a great screen but a bad battery .",
    "this laptop has a poor screen but a great battery .",
    "the price is high but the quality is great .",
    "the price is low but the quality is bad .",
    "a good phone with a bright screen and a long battery life .",
    "a bad phone with a dim screen and a short battery life .",
    "she said the phone was great and the battery was good .",
    "he said the phone was bad and the battery was poor .",
    "the battery charges fast and the screen looks great .",
    "the battery drains fast and the screen looks bad .",
]


# ---------------------------------------------------------------------------
# 一、数据预处理
# ---------------------------------------------------------------------------
def tokenize(text: str) -> list[str]:
    """归一化 → 去标签 → 小写 → 正则分词（保留否定词与英文缩写）。"""
    text = unicodedata.normalize("NFKC", html.unescape(text)).replace("’", "'")
    text = re.sub(r"<[^>]*>", " ", text).lower()
    return re.findall(PREPROCESSING["token_pattern"], text)


def read_reviews(path: Path) -> list[tuple[int, list[str]]]:
    """读取 Amazon 评论 CSV（label,title,text），返回 (情感标签, token 列表)。"""
    reviews: list[tuple[int, list[str]]] = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        first = True
        for row in reader:
            if not row or not any(cell.strip() for cell in row):
                continue
            fields = [cell.strip().lower() for cell in row]
            if first and fields[:1] in (["label"], ["polarity"]):
                first = False
                continue
            first = False
            if len(row) != 3 or row[0].strip() not in {"1", "2"}:
                raise ValueError(f"{path.name} 第 {reader.line_num} 行格式错误，应为 label,title,text")
            tokens = tokenize(row[1] + " " + row[2])
            if not tokens:
                raise ValueError(f"{path.name} 第 {reader.line_num} 行正文为空")
            reviews.append((int(row[0]), tokens))
    return reviews


def load_corpus(csv_path: Path, demo: bool) -> tuple[list[list[str]], list[int], str]:
    """返回 (分词语料, 标签, 数据来源说明)。"""
    if demo:
        corpus = [tokenize(sentence) for sentence in DEMO_SENTENCES]
        return corpus, [2] * len(corpus), f"内置演示语料（{len(corpus)} 句）"
    if not csv_path.exists():
        raise SystemExit(
            f"找不到语料文件：{csv_path}\n"
            "请用 --csv 指定训练集路径，或加 --demo 用内置小语料演示流程。"
        )
    reviews = read_reviews(csv_path)
    if not reviews:
        raise SystemExit(f"语料为空：{csv_path}")
    return [tokens for _, tokens in reviews], [label for label, _ in reviews], str(csv_path)


# ---------------------------------------------------------------------------
# 二、词表构建与受控训练
# ---------------------------------------------------------------------------
def build_vocab(corpus: list[list[str]], min_count: int) -> Counter:
    """统计词频并打印 min_count 截断情况。"""
    counts = Counter(word for sentence in corpus for word in sentence)
    kept = {word: count for word, count in counts.items() if count >= min_count}
    print(f"去重词型（distinct）   : {len(counts):,}")
    print(f"min_count={min_count} 过滤后词表 : {len(kept):,}")
    print(f"被丢弃的长尾词         : {len(counts) - len(kept):,}")
    return counts


def train_model(corpus, window: int, vector_size: int, epochs: int, min_count: int, seed: int):
    """训练单个 CBOW 模型，返回 (模型, 耗时秒数)。除 window 外参数全部固定。"""
    from gensim.models import Word2Vec

    started = time.perf_counter()
    model = Word2Vec(
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        seed=seed,
        **FIXED_PARAMETERS,
    )
    model.build_vocab(corpus)
    model.train(corpus, total_examples=model.corpus_count, epochs=epochs, compute_loss=True)
    elapsed = time.perf_counter() - started
    return model, elapsed


# ---------------------------------------------------------------------------
# 三、相似度实测与结果导出
# ---------------------------------------------------------------------------
def report_pairs(models: dict[int, object]) -> list[dict]:
    """输出 5 组词对在两个窗口下的余弦相似度。"""
    print("\n5 组指定词对的余弦相似度")
    print(f"{'词对':<22}{'window=2':>10}{'window=5':>10}{'差值':>10}")
    rows = []
    for left, right in PAIRS:
        values = {}
        for window, model in models.items():
            wv = model.wv
            values[window] = float(wv.similarity(left, right)) if (left in wv and right in wv) else None
        pair = f"{left} - {right}"
        if values[2] is None or values[5] is None:
            print(f"{pair:<22}{'OOV':>10}{'OOV':>10}{'-':>10}")
        else:
            print(f"{pair:<22}{values[2]:>10.4f}{values[5]:>10.4f}{values[5] - values[2]:>+10.4f}")
        rows.append({"pair": pair, "window_2": values[2], "window_5": values[5]})
    return rows


def report_neighbors(models: dict[int, object], topn: int = 5) -> None:
    """输出指定词的 Top-N 近邻；未登录词（OOV）明确标注。"""
    print(f"\n指定词的 Top-{topn} 语义近邻")
    for word in QUERIES:
        print(f"[{word}]")
        for window, model in models.items():
            wv = model.wv
            if word not in wv:
                print(f"  window={window}: OOV（词频低于 min_count，未进入词表）")
                continue
            neighbors = "，".join(f"{near} {score:.3f}" for near, score in wv.most_similar(word, topn=topn))
            print(f"  window={window}: {neighbors}")


def export_handoff(model, folder: Path, seed: int) -> None:
    """导出实验 2 需要的词向量矩阵与词表：PAD 全零，UNK 为固定随机小向量。"""
    import numpy as np

    folder.mkdir(parents=True, exist_ok=True)
    words = ["<PAD>", "<UNK>"] + model.wv.index_to_key
    matrix = np.zeros((len(words), model.vector_size), dtype=np.float32)
    matrix[1] = np.random.default_rng(seed).normal(0, 0.05, model.vector_size)
    matrix[2:] = model.wv.vectors

    np.save(folder / "embedding.npy", matrix, allow_pickle=False)
    vocab = {
        "pad_id": 0,
        "unk_id": 1,
        "id_to_token": words,
        "token_to_id": {word: index for index, word in enumerate(words)},
        "vector_size": model.vector_size,
    }
    (folder / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8")
    (folder / "preprocessing.json").write_text(
        json.dumps(PREPROCESSING, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    reloaded = np.load(folder / "embedding.npy", allow_pickle=False)
    assert reloaded.shape == (len(words), model.vector_size), "矩阵行数与词表长度不一致"
    assert not reloaded[0].any(), "PAD 行必须全零"
    print(f"\n已导出交接文件：{folder}")
    print(f"  embedding.npy : {reloaded.shape} {reloaded.dtype}")
    print(f"  vocab.json    : {len(words):,} 个 token（含 PAD/UNK），维度 {model.vector_size}")


# ---------------------------------------------------------------------------
# 四、主流程
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="实验 1：词向量表示与相似度分析")
    parser.add_argument("--csv", type=Path, default=Path("data/train_sample.csv"), help="训练语料 CSV 路径")
    parser.add_argument("--out", type=Path, default=Path("outputs"), help="输出目录")
    parser.add_argument("--demo", action="store_true", help="使用内置小语料演示流程")
    parser.add_argument("--vector-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--min-count", type=int, default=None, help="默认 2；--demo 时为 1")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        import gensim  # noqa: F401  提前检查依赖，给出友好提示
    except ImportError:
        raise SystemExit("缺少依赖：请先执行  pip install gensim numpy")

    # gensim 的初始化会用到 Python 的字符串哈希，重设 PYTHONHASHSEED 后才能逐位复现
    if os.environ.get("PYTHONHASHSEED") != str(args.seed):
        env = dict(os.environ, PYTHONHASHSEED=str(args.seed))
        return subprocess.call([sys.executable, *sys.argv], env=env)

    min_count = args.min_count if args.min_count is not None else (1 if args.demo else 2)
    corpus, labels, source = load_corpus(args.csv, args.demo)
    token_total = sum(len(sentence) for sentence in corpus)

    print("=" * 68)
    print("实验 1 · 词向量表示与相似度分析")
    print("=" * 68)
    print(f"数据来源 : {source}")
    print(f"文档数量 : {len(corpus):,}")
    print(f"总 token : {token_total:,}")
    if not args.demo:
        negative = sum(1 for label in labels if label == 1)
        positive = sum(1 for label in labels if label == 2)
        print(f"标签分布 : 负面(1) {negative:,} / 正面(2) {positive:,}")
    print(f"超参数   : CBOW, vector_size={args.vector_size}, epochs={args.epochs}, "
          f"min_count={min_count}, seed={args.seed}, workers=1")

    build_vocab(corpus, min_count)

    models: dict[int, object] = {}
    print("\n受控训练：只改变 window")
    for window in WINDOWS:
        model, seconds = train_model(corpus, window, args.vector_size, args.epochs, min_count, args.seed)
        models[window] = model
        print(f"  window={window}: 词表 {len(model.wv):,} 个词，耗时 {seconds:.2f}s，"
              f"训练损失 {model.get_latest_training_loss():.0f}")

    pairs = report_pairs(models)
    report_neighbors(models)

    export_handoff(models[5], args.out / "window_5", args.seed)
    out_csv = args.out / "word_pairs.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["pair", "window_2", "window_5"])
        for row in pairs:
            writer.writerow([row["pair"], row["window_2"], row["window_5"]])
    print(f"  词对相似度    : {out_csv}")

    print("\n完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
