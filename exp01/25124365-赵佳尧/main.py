# -*- coding: utf-8 -*-
"""实验1 核心代码：词向量表示与相似度分析（窗口 2 与 5 对照）。

用法：
    python main.py --train-csv data/train_sample.csv --output-dir outputs/run

流程：
    1) 读取 Amazon 评论 CSV（label,title,text；label 1=负面 2=正面）
    2) 统一预处理（NFKC → 去 HTML 标签 → 小写 → 正则分词，保留否定词与缩写）
    3) 固定除窗口外的全部超参数，分别训练 window=2 与 window=5 两个 CBOW 模型
    4) 计算 5 组指定词对余弦相似度、10 个查询词的 Top-5 近邻
    5) 取 200 个高频词做 PCA 二维投影（探索性观察）
    6) 导出 window=5 的词向量矩阵与词表，供实验2 的 TextCNN 使用

本文件自包含，只依赖 gensim / numpy / scikit-learn / matplotlib。
"""
import argparse
import csv
import hashlib
import html
import json
import os
import platform
import re
import subprocess
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------- 预处理契约
PREPROCESSING = {
    "version": 1,
    "language": "English",
    "unicode": "NFKC",
    "lowercase": True,
    "html_unescape": True,
    "remove_html_tags": True,
    "token_pattern": r"[a-z]+(?:'[a-z]+)?|[0-9]+",
    "remove_stopwords": False,   # 下游是情感分类，保留否定词（not/no/never）
    "combine": "title + space + text",
}


def tokenize(text):
    """与实验2 共用同一版预处理，保证词表 id 不错位。"""
    text = unicodedata.normalize("NFKC", html.unescape(text)).replace("\u2019", "'")
    text = re.sub(r"<[^>]*>", " ", text)
    return re.findall(PREPROCESSING["token_pattern"], text.lower())


def read_reviews(path):
    """读取 Amazon 评论 CSV：支持有表头与无表头，校验标签与空文本。"""
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        first = True
        for row in reader:
            if not row or not any(s.strip() for s in row):
                continue
            fields = [s.strip().lower() for s in row]
            if first and fields in (["label", "title", "text"], ["polarity", "title", "text"],
                                    ["label", "title", "content"]):
                first = False
                continue
            first = False
            if len(row) != 3 or row[0].strip() not in {"1", "2"}:
                raise ValueError(f"{path}: 第 {reader.line_num} 行格式错误，应为 label(1/2),title,text")
            tokens = tokenize(row[1] + " " + row[2])
            if not tokens:
                raise ValueError(f"{path}: 第 {reader.line_num} 行没有有效英文文本")
            yield int(row[0]), tokens


def review_key(tokens):
    """按模型可见文本去重/查重（与标签无关）。"""
    return hashlib.sha256(" ".join(tokens).encode()).digest()


def exclusion_keys(paths):
    return {review_key(tokens) for path in paths or [] for _, tokens in read_reviews(path)}


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_csv(path, fields, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- 实验参数
# 除 window 外全部锁死，保证"只有一个变量"
FIXED = {"workers": 1, "shrink_windows": False, "alpha": 0.025, "min_alpha": 0.0001,
         "negative": 5, "hs": 0, "sample": 1e-3}
PAIRS = [("good", "great"), ("good", "bad"), ("phone", "battery"),
         ("expensive", "cheap"), ("screen", "battery")]
QUERIES = ["good", "bad", "great", "phone", "battery", "screen", "price", "quality", "easy", "poor"]


def export_handoff(model, folder, seed, vector_size):
    """导出实验2 需要的矩阵与词表：行号 = 词 id，PAD=0 全零，UNK=1 固定随机向量。"""
    import numpy as np
    words = ["<PAD>", "<UNK>"] + list(model.wv.index_to_key)
    matrix = np.zeros((len(words), vector_size), dtype=np.float32)
    matrix[1] = np.random.default_rng(seed).normal(0, 0.05, vector_size)   # UNK
    matrix[2:] = model.wv.vectors
    np.save(folder / "embedding.npy", matrix, allow_pickle=False)
    write_json(folder / "vocab.json", {
        "pad_id": 0, "unk_id": 1, "id_to_token": words,
        "token_to_id": {w: i for i, w in enumerate(words)},
        "vector_size": vector_size,
    })
    write_json(folder / "preprocessing.json", PREPROCESSING)


def project(words, model, method, seed, folder):
    """200 个高频词的二维投影：仅作探索性观察，两模型的轴不可跨图比较。"""
    import numpy as np
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("跳过绘图：未安装 matplotlib")
        return
    x = model.wv[words]
    if method == "tsne":
        from sklearn.manifold import TSNE
        xy = TSNE(n_components=2, perplexity=min(30, max(1, (len(words) - 1) / 3)),
                  random_state=seed, init="pca", learning_rate="auto").fit_transform(x)
    else:
        from sklearn.decomposition import PCA
        xy = PCA(n_components=2, svd_solver="full").fit_transform(x)
    write_csv(folder / "projection.csv", ["word", "x", "y"],
              [{"word": w, "x": float(v[0]), "y": float(v[1])} for w, v in zip(words, xy)])
    fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
    ax.scatter(xy[:, 0], xy[:, 1], s=20, alpha=0.65)
    for w, (px, py) in zip(words[:30], xy[:30]):
        ax.annotate(w, (px, py), xytext=(3, 3), textcoords="offset points", fontsize=8)
    ax.set_title(f"{method.upper()} | window={model.window} | {len(words)} words (30 labelled)")
    ax.set_xlabel("Projection axis 1")
    ax.set_ylabel("Projection axis 2")
    ax.text(0, -0.1, "Exploratory projection; axes of separately fitted models are not aligned.",
            transform=ax.transAxes, fontsize=9)
    fig.savefig(folder / f"{method}.png", dpi=160)
    plt.close(fig)


def run(args):
    import numpy as np
    from gensim.models import Word2Vec

    rows = list(read_reviews(args.train_csv))
    if len(rows) < 10:
        raise ValueError("训练集太小，至少需要 10 条评论")
    keys = [review_key(t) for _, t in rows]
    if len(set(keys)) != len(keys):
        raise ValueError(f"训练集存在 {len(keys) - len(set(keys))} 条重复文本，请先去重")
    overlap = len(set(keys) & exclusion_keys(args.exclude_csv))
    if overlap:
        raise ValueError(f"{overlap} 条训练文本同时出现在开发/测试集，存在数据泄漏")

    corpus = [t for _, t in rows]
    counts = Counter(w for s in corpus for w in s)
    words = sorted((w for w in counts if counts[w] >= args.min_count),
                   key=lambda w: (-counts[w], w))[:args.top_words]
    if len(words) < 10:
        raise ValueError("滤波后词表过小，请降低 --min-count 或增大语料")

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError(f"输出目录非空：{args.output_dir}（换一个新目录，避免覆盖历史结果）")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    queries = args.queries
    manifest = {
        "purpose": "lab1 word2vec window=2 vs 5",
        "train_name": Path(args.train_csv).name,
        "train_sha256": sha256_file(args.train_csv),
        "rows": len(rows), "labels": dict(Counter(l for l, _ in rows)),
        "token_count": sum(map(len, corpus)),
        "vocab_after_min_count": len(words),
        "min_count": args.min_count, "vector_size": args.vector_size, "epochs": args.epochs,
        "seed": args.seed, "fixed_parameters": FIXED,
        "excluded_files": [{"name": Path(p).name, "sha256": sha256_file(p)} for p in args.exclude_csv],
        "overlap_count": overlap,
        "python": platform.python_version(),
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "preprocessing": PREPROCESSING,
        "label_encoding": {"1": "negative", "2": "positive"},
    }
    write_json(args.output_dir / "run_manifest.json", {**manifest, "status": "running"})

    comparison, pairs, neighbours = [], [], []
    models = {}
    for window in (2, 5):
        folder = args.output_dir / f"window_{window}"
        folder.mkdir()
        started = time.perf_counter()
        model = Word2Vec(vector_size=args.vector_size, window=window, min_count=args.min_count,
                         sg=0, seed=args.seed, **FIXED)
        model.build_vocab(corpus)
        model.train(corpus, total_examples=model.corpus_count, epochs=args.epochs, compute_loss=True)
        seconds = time.perf_counter() - started
        model.save(str(folder / "word2vec.model"))
        model.wv.save(str(folder / "vectors.kv"))
        export_handoff(model, folder, args.seed, args.vector_size)
        if args.projection != "none":
            project(words, model, args.projection, args.seed, folder)
        models[window] = model
        comparison.append({"window": window, "vocab_size": len(model.wv), "rows": len(rows),
                           "train_seconds": round(seconds, 2),
                           "training_loss": round(float(model.get_latest_training_loss()), 1)})
        for a, b in PAIRS:
            ok = a in model.wv and b in model.wv
            pairs.append({"window": window, "word1": a, "word2": b,
                          "similarity": round(float(model.wv.similarity(a, b)), 4) if ok else "",
                          "status": "ok" if ok else "oov"})
        for q in queries:
            if q not in model.wv:
                neighbours.append({"window": window, "query": q, "rank": "", "neighbor": "",
                                   "similarity": "", "status": "oov"})
                continue
            for rank, (near, score) in enumerate(model.wv.most_similar(q, topn=5), 1):
                neighbours.append({"window": window, "query": q, "rank": rank, "neighbor": near,
                                   "similarity": round(float(score), 4), "status": "ok"})
        print(f"window={window}: {len(model.wv)} words, {seconds:.2f}s")

    write_csv(args.output_dir / "comparison.csv", list(comparison[0]), comparison)
    write_csv(args.output_dir / "word_pairs.csv", list(pairs[0]), pairs)
    write_csv(args.output_dir / "neighbors.csv",
              ["window", "query", "rank", "neighbor", "similarity", "status"], neighbours)

    # 窗口差异量化：200 个高频词的成对余弦相似度的 Spearman 相关
    stats = {}
    try:
        from scipy.stats import spearmanr
        import numpy as np
        def sim_matrix(model):
            v = np.array([model.wv[w] for w in words], dtype=np.float64)
            v /= np.linalg.norm(v, axis=1, keepdims=True)
            s = v @ v.T
            return s[np.triu_indices(len(words), 1)]
        s2, s5 = sim_matrix(models[2]), sim_matrix(models[5])
        delta = np.abs(s2 - s5)
        stats = {"pairs": int(len(s2)),
                 "spearman": round(float(spearmanr(s2, s5).correlation), 4),
                 "mean_abs_delta": round(float(delta.mean()), 4),
                 "pct_delta_gt_0.1": round(float((delta > 0.1).mean() * 100), 2),
                 "max_abs_delta": round(float(delta.max()), 4)}
    except Exception as error:            # 统计失败不影响主要产物
        stats = {"error": str(error)}

    write_json(args.output_dir / "run_manifest.json", {**manifest, "status": "complete", "window_delta": stats})

    print("\n== 词对相似度 ==")
    print(f"{'pair':<22}{'w2':>8}{'w5':>8}")
    by_pair = {}
    for r in pairs:
        by_pair.setdefault((r["word1"], r["word2"]), {})[r["window"]] = r["similarity"]
    for (a, b), d in by_pair.items():
        print(f"{a + '-' + b:<22}{d.get(2, ''):>8}{d.get(5, ''):>8}")
    print("\n== 窗口差异 ==", stats)
    print(f"\n完成：结果在 {args.output_dir}；实验2 交接文件在 {args.output_dir / 'window_5'}")


def main():
    p = argparse.ArgumentParser(description="实验1：Word2Vec 窗口对照与相似度分析")
    p.add_argument("--train-csv", type=Path, default=None,
                   help="训练集 CSV（label,title,text）；缺省时自动查找 examples/demo_train.csv")
    p.add_argument("--output-dir", type=Path, default=Path("outputs/run"))
    p.add_argument("--exclude-csv", type=Path, nargs="*", default=[],
                   help="开发/测试集，只用于查重，绝不参与训练")
    p.add_argument("--vector-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--min-count", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--top-words", type=int, default=200, help="参与投影/统计的高频词数")
    p.add_argument("--projection", choices=["pca", "tsne", "none"], default="pca")
    p.add_argument("--queries", nargs="*", default=QUERIES, help="查询 Top-5 近邻的词")
    args = p.parse_args()

    if args.train_csv is None:                     # 不给参数时，按常见位置自动查找演示数据
        for candidate in (Path("examples/demo_train.csv"), Path("data/train_sample.csv"),
                          Path("../../2026-autumn/exp01-word2vec/examples/demo_train.csv")):
            if candidate.exists():
                args.train_csv = candidate
                print(f"未指定 --train-csv，自动使用 {candidate}")
                break
        else:
            p.error("缺少训练数据。请使用：python main.py --train-csv 你的/train.csv --output-dir outputs/run")

    # gensim 初始化依赖进程级字符串哈希，固定后才能在相同环境逐位复现
    if os.environ.get("PYTHONHASHSEED") != str(args.seed):
        env = dict(os.environ, PYTHONHASHSEED=str(args.seed))
        return subprocess.call([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]], env=env)
    try:
        run(args)
    except (ValueError, OSError) as error:
        sys.exit(f"实验失败：{error}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
