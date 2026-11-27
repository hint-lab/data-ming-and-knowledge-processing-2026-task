"""实验1：词向量表示与相似度分析

做什么
------
用同一份 Amazon 评论训练集，训练两个 Word2Vec 模型。
两者唯一的区别是滑动窗口 window=2 与 window=5，其余全部固定，
然后对比：词对余弦相似度、Top-5 近邻、二维投影图，
最后把 window=5 的词向量导出，交给实验2 的 TextCNN 使用。

受控条件（除 window 外全部锁死）
------------------------------
向量维度 128 / 训练轮数 10 / 随机种子 42 / CBOW / min_count=2 / 单线程

用法
----
    python main.py                            跑完整实验
    python main.py --live good bad battery    现场查词
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from gensim.models import KeyedVectors, Word2Vec

from lab_common import read_reviews, sha256

# ===== 受控参数：除窗口外全部固定，保证唯一变量 =====
VECTOR_SIZE = 128        # 每个词用几个数字表示
EPOCHS = 10              # 整份语料训练几遍
SEED = 42                # 随机种子，固定后结果可复现
MIN_COUNT = 2            # 出现少于 2 次的词丢弃
ALGORITHM = "cbow"       # cbow = 看周围猜中间
WINDOWS = (2, 5)         # 唯一变量

# 其余所有训练参数也固定下来，避免任何隐藏差异
FIXED_ARGS = {
    "workers": 1,             # 单线程：多线程更新顺序不确定，结果不可复现
    "shrink_windows": False,  # 不随机缩小窗口，让两种设定差异干净可解释
    "alpha": 0.025,           # 初始学习率
    "min_alpha": 0.0001,      # 最终学习率
    "negative": 5,            # 负采样个数
    "hs": 0,                  # 不用层次 softmax
    "sample": 1e-3,           # 高频词降采样阈值
}

# 老师指定的 5 组固定词对
WORD_PAIRS = [
    ("good", "great"),       # 近义词：直觉应该高
    ("good", "bad"),         # 反义词：直觉应该低，实测往往很高
    ("phone", "battery"),    # 同主题：应该高
    ("expensive", "cheap"),  # 反义词：应该低
    ("screen", "battery"),   # 同主题：应该高
]

# 老师指定的 5 个查询词
QUERY_WORDS = ["good", "bad", "great", "phone", "battery"]


def load_corpus(csv_path):
    """读取训练语料，切成词列表。

    数据红线：拒绝用 dev/test 训练，防止测试信息泄漏进词向量。
    """
    stem = Path(csv_path).stem.lower()
    if stem in {"dev", "test", "valid", "validation"}:
        raise ValueError("禁止用开发集/测试集训练词向量")

    # read_reviews 解析 CSV，tokens() 分词（小写、去标签、保留否定词）
    corpus = [review.tokens() for review in read_reviews(csv_path)]
    if not corpus:
        raise ValueError("训练数据为空")
    return corpus


def train_one(corpus, window):
    """训练一个指定窗口的模型，返回模型和耗时。

    build_vocab 和 train 必须分两步：
      build_vocab 先数一遍词频，决定哪些词进词表（受 min_count 控制）
      train 再真正更新向量
    所以两个模型的词表完全相同，但向量完全不同。
    """
    start = time.perf_counter()
    model = Word2Vec(
        vector_size=VECTOR_SIZE,
        window=window,
        min_count=MIN_COUNT,
        sg=int(ALGORITHM == "skipgram"),   # 0=CBOW, 1=Skip-gram
        seed=SEED,
        **FIXED_ARGS,
    )
    model.build_vocab(corpus)
    model.train(
        corpus,
        total_examples=model.corpus_count,
        epochs=EPOCHS,
        compute_loss=True,   # 记录训练损失，写进对照表
    )
    return model, time.perf_counter() - start


def export_handoff(model, folder):
    """导出实验2 需要的四个文件。

    embedding.npy 的行号就是词ID，必须和 vocab.json 严格对应。
    第 0 行 PAD 全零（补句子长度用），
    第 1 行 UNK 是固定随机向量（词表外词用），
    从第 2 行开始才是真正训练出来的词向量。
    """
    folder.mkdir(parents=True, exist_ok=True)

    words = ["<PAD>", "<UNK>"] + model.wv.index_to_key
    matrix = np.zeros((len(words), model.vector_size), dtype=np.float32)
    # UNK 用固定种子生成，保证每次导出都一样、可复现
    matrix[1] = np.random.default_rng(SEED).normal(0, 0.05, model.vector_size)
    # 第 2 行起，逐行搬运训练好的向量
    matrix[2:] = model.wv.vectors

    np.save(folder / "embedding.npy", matrix, allow_pickle=False)

    vocab = {
        "pad_id": 0,
        "unk_id": 1,
        "id_to_token": words,                                 # 行号 -> 词
        "token_to_id": {w: i for i, w in enumerate(words)},    # 词 -> 行号
        "vector_size": model.vector_size,
    }
    (folder / "vocab.json").write_text(
        json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8")

    # 完整模型（可继续训练）与只读向量库（查词更快更小）
    model.save(str(folder / "word2vec.model"))
    model.wv.save(str(folder / "vectors.kv"))

    return matrix.shape


def draw_pca(model, out_path, n_words=200):
    """把 128 维向量压成 2 维画散点图。

    这只是"看一眼"，不能用来下定量结论：
    两张图是各自独立拟合的，坐标轴没有对齐。
    """
    try:
        from matplotlib import use as mpl_use
        from matplotlib import pyplot as plt
        from sklearn.decomposition import PCA
    except ImportError:
        print("  跳过绘图：缺少 matplotlib 或 scikit-learn")
        return

    words = model.wv.index_to_key[:n_words]
    xy = PCA(n_components=2, svd_solver="full").fit_transform(model.wv[words])

    mpl_use("Agg")  # 无界面后端，直接存文件
    fig, ax = plt.subplots(figsize=(9, 6.5), constrained_layout=True)
    ax.scatter(xy[:, 0], xy[:, 1], s=18, alpha=0.6, color="#285887")
    for word, (x, y) in zip(words[:30], xy[:30]):
        ax.annotate(word, (x, y), xytext=(3, 3), textcoords="offset points", fontsize=8)
    ax.set_title(f"PCA projection | window={model.window} | {len(words)} words")
    ax.text(0, -0.08, "Exploratory only; axes of separately fitted models are not aligned.",
            transform=ax.transAxes, fontsize=8)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def run(train_csv, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("步骤1 · 语料准备")
    corpus = load_corpus(train_csv)
    counts = {}
    for sentence in corpus:
        for word in sentence:
            counts[word] = counts.get(word, 0) + 1

    total_tokens = sum(len(s) for s in corpus)
    kept_tokens = sum(v for v in counts.values() if v >= MIN_COUNT)
    print(f"  训练文档数       : {len(corpus)}")
    print(f"  原始 token 总数  : {total_tokens}")
    print(f"  词型数（去重）   : {len(counts)}")
    print(f"  min_count 过滤后 : {kept_tokens} 个 token 保留")
    print(f"  数据指纹 SHA256  : {sha256(train_csv)[:16]}…")

    print("\n步骤2 · 受控训练（唯一变量 window）")
    print(f"  固定: dim={VECTOR_SIZE}, epochs={EPOCHS}, seed={SEED}, "
          f"algo={ALGORITHM}, min_count={MIN_COUNT}")

    models, seconds, losses = {}, {}, {}
    for window in WINDOWS:
        models[window], seconds[window] = train_one(corpus, window)
        losses[window] = models[window].get_latest_training_loss()
        print(f"  window={window}: 词表 {len(models[window].wv)} 词，"
              f"耗时 {seconds[window]:.2f}s，损失 {losses[window]:.0f}")

    print("\n步骤3 · 相似度实测")
    print(f"  {'词对':<20}{'window=2':>10}{'window=5':>10}{'变化':>10}")
    with open(out_dir / "word_pairs.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["词对", "window=2", "window=5", "变化"])
        for left, right in WORD_PAIRS:
            values = []
            for window in WINDOWS:
                wv = models[window].wv
                values.append(wv.similarity(left, right)
                              if left in wv and right in wv else None)
            name = f"{left}-{right}"
            if None in values:
                print(f"  {name:<20}{'oov':>10}{'oov':>10}")
                writer.writerow([name, "oov", "oov", "-"])
            else:
                print(f"  {name:<20}{values[0]:>10.4f}{values[1]:>10.4f}"
                      f"{values[1] - values[0]:>+10.4f}")
                writer.writerow([name, f"{values[0]:.4f}", f"{values[1]:.4f}",
                                 f"{values[1] - values[0]:+.4f}"])

    print("\n  Top-5 近邻对比")
    with open(out_dir / "neighbors.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["查询词", "window=2 Top5", "window=5 Top5"])
        for word in QUERY_WORDS:
            cells = []
            for window in WINDOWS:
                wv = models[window].wv
                cells.append(", ".join(n for n, _ in wv.most_similar(word, topn=5))
                             if word in wv else "OOV")
            print(f"    {word:<10}w2: {cells[0]}")
            print(f"    {'':<10}w5: {cells[1]}")
            writer.writerow([word, cells[0], cells[1]])

    print("\n步骤4 · 导出与降维观察")
    shape = export_handoff(models[5], out_dir / "handoff")
    print(f"  window=5 词向量矩阵 {shape}（行号=词ID）→ {out_dir / 'handoff'}")
    draw_pca(models[5], out_dir / "pca_window5.png")
    print(f"  PCA 图 → {out_dir / 'pca_window5.png'}")

    print("\n完成。结果文件在：", out_dir)
    print("=" * 60)


def live_query(vectors, words):
    """现场查词：老师给一个新词，立刻查出近邻并解释。

    词不在词表里（OOV）时，如实说"查表模型没有这个词的向量"，
    不能编造结果——缺词就是缺词。
    """
    kv = KeyedVectors.load(str(vectors))
    print(f"词表大小 {len(kv)}，维度 {kv.vector_size}\n")
    for raw in words:
        word = raw.lower().strip()
        if word not in kv:
            print(f"{word}: OOV —— 不在词表内，普通 Word2Vec 无法为它生成向量。")
            continue
        near = kv.most_similar(word, topn=5)
        print(f"{word}: " + ", ".join(f"{n}({s:.3f})" for n, s in near))


def main():
    parser = argparse.ArgumentParser(description="实验1 词向量对照实验")
    parser.add_argument("--train-csv", default="data/train_sample.csv")
    parser.add_argument("--output-dir", default="outputs/final")
    parser.add_argument("--live", nargs="*", help="现场查词模式")
    parser.add_argument("--vectors", default="outputs/final/handoff/vectors.kv")
    args = parser.parse_args()

    if args.live is not None:
        live_query(args.vectors, args.live)
        return
    run(args.train_csv, args.output_dir)


if __name__ == "__main__":
    main()
