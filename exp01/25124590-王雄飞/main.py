"""实验一 词向量表示与相似度分析 —— 核心代码

做了什么
--------
1. 读入训练集 CSV（Amazon Polarity 格式：label,title,text，label 1=负面 2=正面）；
2. 用同一份语料、同一随机种子、同一批固定参数训练两个 Word2Vec 模型：window=2 与 window=5；
3. 输出词对相似度表、10 个查询词的 Top-5 近邻表、两窗口对照表；
4. 画一张"相关度图"：左图是 32 个代表词的二维位置（颜色=语义组），右图是去均值余弦热图；
5. 导出 window=5 的 embedding.npy / vocab.json，供实验二 TextCNN 使用。

运行
----
    python main.py --train-csv data/train_sample.csv --out-dir outputs

依赖：gensim, numpy, scipy, scikit-learn, matplotlib
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import logging
import os
import re
import subprocess
import sys
import time
import unicodedata
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------- 预处理契约
# 与实验二共用同一套规则：转小写、HTML 反转义、去标签、保留否定词与英文缩写。
TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?|[0-9]+")
TAG_RE = re.compile(r"<[^>]*>")

FIXED = {"workers": 1, "shrink_windows": False, "alpha": 0.025,
         "min_alpha": 1e-4, "negative": 5, "hs": 0, "sample": 1e-3}

QUERIES = ["good", "bad", "great", "phone", "battery",
           "screen", "price", "quality", "easy", "poor"]
PAIRS = [("good", "great"), ("good", "bad"), ("phone", "battery"),
         ("expensive", "cheap"), ("screen", "battery")]

# 人工划分的语义组：用来检验"相关的词是不是挨在一起"。
GROUPS = {
    "positive sentiment": ["good", "great", "best", "love", "excellent",
                           "wonderful", "perfect", "nice"],
    "negative sentiment": ["bad", "worst", "poor", "terrible", "awful",
                           "hate", "boring", "disappointing"],
    "books & film": ["book", "movie", "film", "cd", "music", "album",
                     "story", "author"],
    "money & value": ["price", "cheap", "expensive", "money", "value",
                      "worth", "cost", "recommend"],
}
COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]


def tokenize(text: str) -> list[str]:
    """lowercase -> strip HTML -> keep letters/numbers/apostrophes."""
    text = unicodedata.normalize("NFKC", html.unescape(str(text))).replace("\u2019", "'")
    return TOKEN_RE.findall(TAG_RE.sub(" ", text).lower())


def read_reviews(path: Path):
    """Yield (label, title, text) from an Amazon CSV, with or without a header."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        first = True
        for row in reader:
            if not row or not any(field.strip() for field in row):
                continue
            header = [field.strip().lower() for field in row]
            if first and header in (["label", "title", "text"],
                                    ["polarity", "title", "text"],
                                    ["label", "title", "content"]):
                first = False
                continue
            first = False
            if len(row) != 3 or row[0].strip() not in {"1", "2"}:
                raise ValueError(f"{path}: expected label 1/2,title,text "
                                 f"(line {reader.line_num})")
            yield int(row[0]), row[1].strip(), row[2].strip()


def text_key(title: str, text: str) -> str:
    """Model-visible text fingerprint, used to detect duplicates."""
    return hashlib.sha256(" ".join(tokenize(title + " " + text)).encode()).hexdigest()


def load_corpus(path: Path, exclude: list[Path]):
    """Return corpus/labels/frame-ish rows after dedupe and overlap check."""
    rows, seen = [], set()
    for label, title, text in read_reviews(path):
        key = text_key(title, text)
        if key in seen:
            raise ValueError("训练集里有重复文本，请先用 prepare_data.py 去重")
        seen.add(key)
        rows.append((label, title, text))
    for other in exclude:
        overlap = seen & {text_key(t, x) for _, t, x in read_reviews(other)}
        if overlap:
            raise ValueError(f"训练集与 {other.name} 有 {len(overlap)} 条重叠（数据泄漏）")
    corpus = [tokenize(title + " " + text) for _, title, text in rows]
    if sum(len(tokens) for tokens in corpus) == 0:
        raise ValueError("训练集没有可用的英文词")
    return corpus, np.array([row[0] for row in rows])


# ------------------------------------------------------------------- 训练
def train(corpus, window: int, vector_size: int, epochs: int,
          min_count: int, seed: int):
    from gensim.models import Word2Vec

    started = time.perf_counter()
    model = Word2Vec(vector_size=vector_size, window=window, min_count=min_count,
                     sg=0, seed=seed, **FIXED)
    model.build_vocab(corpus)
    model.train(corpus, total_examples=model.corpus_count, epochs=epochs,
                compute_loss=True)
    return model, time.perf_counter() - started


def export_handoff(model, folder: Path, seed: int):
    """embedding.npy + vocab.json：行号即词 ID，PAD=0 全零，UNK=1 固定随机。"""
    words = ["<PAD>", "<UNK>"] + model.wv.index_to_key
    matrix = np.zeros((len(words), model.vector_size), dtype=np.float32)
    matrix[1] = np.random.default_rng(seed).normal(0, 0.05, model.vector_size)
    matrix[2:] = model.wv.vectors
    np.save(folder / "embedding.npy", matrix, allow_pickle=False)
    (folder / "vocab.json").write_text(json.dumps({
        "pad_id": 0, "unk_id": 1, "id_to_token": words,
        "token_to_id": {w: i for i, w in enumerate(words)},
        "vector_size": model.vector_size}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")


# --------------------------------------------------------------- 相关度图
def relatedness(model, words, groups, folder: Path, method: str, seed: int):
    """左：二维位置图；右：去均值余弦热图。返回组内/组间统计。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Polygon
    from scipy.spatial import ConvexHull

    matrix = np.vstack([model.wv[w] for w in words]).astype(np.float64)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12
    sim = matrix @ matrix.T

    if method == "tsne":
        from sklearn.manifold import TSNE
        xy = TSNE(n_components=2, perplexity=min(30.0, max(2.0, (len(words) - 1) / 3)),
                  random_state=seed, init="pca", learning_rate="auto").fit_transform(
                      matrix.astype(np.float32))
        note = "t-SNE"
    else:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2, svd_solver="full").fit(matrix)
        xy, note = pca.transform(matrix), f"PCA ({pca.explained_variance_ratio_[0]:.0%})"

    member = {w: name for name, ws in groups.items() for w in ws}
    fig = plt.figure(figsize=(18, 8.5), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=[1.45, 1])
    ax = fig.add_subplot(grid[0, 0])
    handles = []
    for index, (name, members) in enumerate(groups.items()):
        colour = COLORS[index % len(COLORS)]
        idx = [words.index(w) for w in members]
        points = xy[idx]
        ax.scatter(points[:, 0], points[:, 1], s=90, color=colour, alpha=0.9,
                   edgecolor="white", linewidth=0.8, zorder=3)
        for word in members:
            ax.annotate(word, (xy[words.index(word), 0], xy[words.index(word), 1]),
                        xytext=(5, 4), textcoords="offset points", fontsize=9,
                        color=colour, fontweight="bold")
        if len(points) >= 3:
            hull = ConvexHull(points)
            ax.add_patch(Polygon(points[hull.vertices], closed=True, fill=True,
                                 facecolor=colour, alpha=0.10, edgecolor=colour,
                                 linewidth=1.2, zorder=2))
        handles.append(Line2D([], [], marker="o", linestyle="", markersize=9,
                              color=colour, label=name))
    drawn = set()
    for i in range(len(words)):
        j = int(np.argsort(-sim[i])[1])
        key = tuple(sorted((i, j)))
        if key in drawn:
            continue
        drawn.add(key)
        ax.plot([xy[i, 0], xy[j, 0]], [xy[i, 1], xy[j, 1]],
                color="#999999", linewidth=0.7, alpha=0.6, zorder=1)
    ax.set_title(f"word-vector relatedness map | window={model.window} | {note}\n"
                 "colour = semantic group, grey line = nearest neighbour", fontsize=11)
    ax.set_xlabel("component 1")
    ax.set_ylabel("component 2")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.2)

    # 右图：减去本模型平均余弦，否则小语料上整张图都是同一种红。
    mean_similarity = float(sim[~np.eye(len(words), dtype=bool)].mean())
    heat = fig.add_subplot(grid[0, 1])
    centred = sim - mean_similarity
    limit = float(np.abs(centred).max())
    image = heat.imshow(centred, cmap="RdBu_r", vmin=-limit, vmax=limit)
    positions, labels, colours, boundary = [], [], [], 0
    for index, (name, members) in enumerate(groups.items()):
        boundary += len(members)
        heat.axhline(boundary - 0.5, color="black", linewidth=0.9)
        heat.axvline(boundary - 0.5, color="black", linewidth=0.9)
        for position in range(boundary - len(members), boundary):
            positions.append(position)
            labels.append(words[position])
            colours.append(COLORS[index % len(COLORS)])
    heat.set_xticks(positions, labels, rotation=90, fontsize=8)
    heat.set_yticks(positions, labels, fontsize=8)
    for tick, colour in zip([*heat.get_xticklabels(), *heat.get_yticklabels()],
                            colours + colours):
        tick.set_color(colour)
    heat.set_title(f"cosine similarity minus model mean ({mean_similarity:+.3f})\n"
                   "red = more related than average, blue = less", fontsize=9)
    fig.colorbar(image, ax=heat, shrink=0.75, label="cosine similarity minus mean")
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               fontsize=10, frameon=False)
    fig.suptitle("Related words sit close in embedding space", fontsize=13)
    fig.savefig(folder / f"relatedness_map_{method}.png", dpi=160)
    plt.close(fig)

    intra = [float(sim[i, j]) for i, j in combinations(range(len(words)), 2)
             if member[words[i]] == member[words[j]]]
    inter = [float(sim[i, j]) for i, j in combinations(range(len(words)), 2)
             if member[words[i]] != member[words[j]]]
    nearest_same = sum(1 for i in range(len(words))
                       if member[words[i]] == member[words[int(np.argsort(-sim[i])[1])]])
    return {"words": len(words), "mean_cosine": round(1 - (1 - mean_similarity), 4),
            "intra": round(float(np.mean(intra)), 4),
            "inter": round(float(np.mean(inter)), 4),
            "gap": round(float(np.mean(intra) - np.mean(inter)), 4),
            "nearest_same_group": f"{nearest_same}/{len(words)}"}


# -------------------------------------------------------------------- 主流程
def main() -> int:
    parser = argparse.ArgumentParser(description="实验一：词向量表示与相似度分析")
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--vector-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20,
                        help="默认 20：5000 条语料上 10 轮还没训开（报告里有对照）")
    parser.add_argument("--min-count", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--projection", choices=["tsne", "pca"], default="tsne")
    parser.add_argument("--exclude-csv", type=Path, nargs="*", default=[],
                        help="开发/测试文件，用于检查数据泄漏")
    args = parser.parse_args()

    # gensim 的初始化依赖 Python 进程级字符串哈希；不固定它就无法逐位复现。
    # 若当前哈希种子与 --seed 不一致，就带新环境把本脚本重新执行一次。
    if os.environ.get("PYTHONHASHSEED") != str(args.seed):
        env = dict(os.environ, PYTHONHASHSEED=str(args.seed))
        return subprocess.call([sys.executable, str(Path(__file__).resolve()),
                                *sys.argv[1:]], env=env)

    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        parser.error(f"{args.out_dir} 非空，换一个目录以免覆盖上次结果")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=args.out_dir / "training.log", level=logging.INFO,
                        format="%(asctime)s %(message)s", force=True)

    print(f"[1/4] 读入 {args.train_csv}")
    corpus, labels = load_corpus(args.train_csv, args.exclude_csv)
    counts = Counter(w for sentence in corpus for w in sentence)
    print(f"      {len(corpus)} 条评论，{sum(counts.values())} 个 token，"
          f"{len(counts)} 种词；标签分布 {dict(zip(*np.unique(labels, return_counts=True)))}")

    results, pairs, neighbors = [], [], []
    for window in (2, 5):
        folder = args.out_dir / f"window_{window}"
        folder.mkdir()
        print(f"[2/4] 训练 window={window}")
        model, seconds = train(corpus, window, args.vector_size, args.epochs,
                               args.min_count, args.seed)
        model.save(str(folder / "word2vec.model"))
        model.wv.save(str(folder / "vectors.kv"))
        export_handoff(model, folder, args.seed)
        print(f"      词表 {len(model.wv)}，耗时 {seconds:.2f}s")
        results.append({"window": window, "vector_size": args.vector_size,
                        "epochs": args.epochs, "vocab_size": len(model.wv),
                        "rows": len(corpus), "train_seconds": round(seconds, 3),
                        "training_loss": model.get_latest_training_loss()})
        for left, right in PAIRS:
            ok = left in model.wv and right in model.wv
            pairs.append({"window": window, "word1": left, "word2": right,
                          "similarity": round(float(model.wv.similarity(left, right)), 4)
                          if ok else "", "status": "ok" if ok else "oov"})
        for word in QUERIES:
            if word not in model.wv:
                neighbors.append({"window": window, "query": word, "rank": "",
                                  "neighbor": "", "similarity": "", "status": "oov"})
                continue
            for rank, (near, score) in enumerate(model.wv.most_similar(word, topn=5), 1):
                neighbors.append({"window": window, "query": word, "rank": rank,
                                  "neighbor": near, "similarity": round(float(score), 4),
                                  "status": "ok"})

        if window == 5:
            words = [w for members in GROUPS.values() for w in members]
            words = [w for w in words if w in model.wv]
            absent = [w for members in GROUPS.values() for w in members
                      if w not in model.wv]
            if absent:
                print(f"      语义组里不在词表的词（已跳过）：{absent}")
            groups = {name: [w for w in members if w in model.wv]
                      for name, members in GROUPS.items()}
            groups = {name: members for name, members in groups.items() if members}
            print(f"[3/4] 画相关度图（{len(words)} 个词，{len(groups)} 组）")
            stats = relatedness(model, words, groups, args.out_dir,
                                args.projection, args.seed)
            print(f"      组内 {stats['intra']} vs 组间 {stats['inter']} "
                  f"(差 {stats['gap']:+.4f})；最近邻同组 {stats['nearest_same_group']}")
            (args.out_dir / "relatedness_stats.json").write_text(
                json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("[4/4] 写结果表")
    def write_table(name, fields, rows):
        with (args.out_dir / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    write_table("comparison.csv", list(results[0]), results)
    write_table("word_pairs.csv", list(pairs[0]), pairs)
    write_table("neighbors.csv", ["window", "query", "rank", "neighbor",
                                  "similarity", "status"], neighbors)
    (args.out_dir / "run_manifest.json").write_text(json.dumps({
        "train_csv": args.train_csv.name,
        "train_sha256": hashlib.sha256(args.train_csv.read_bytes()).hexdigest(),
        "rows": len(corpus), "tokens": sum(counts.values()),
        "labels": {str(k): int(v) for k, v in zip(*np.unique(labels, return_counts=True))},
        "config": {"vector_size": args.vector_size, "epochs": args.epochs,
                   "min_count": args.min_count, "seed": args.seed,
                   "projection": args.projection},
        "fixed_parameters": FIXED, "status": "complete"},
        ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"完成，结果在 {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
