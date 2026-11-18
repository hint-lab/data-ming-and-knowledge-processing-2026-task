# 数据挖掘实验1：Word2Vec与近邻对比
# 基于课程仓库2026实验1参考代码整理；新增功能：compare_neighbors。
# 为便于只提交一个Python文件，将lab_common.py公共函数合并到本文件。
# 依赖：使用原实验目录requirements.txt安装。
# 在原实验目录运行示例（请替换main.py的实际路径）：
# python 路径/main.py --train-csv data/train_sample.csv --output-dir outputs/my_submission --exclude-csv ../../2025-spring/exp02-sentiment-classificationn/dataset/dev.csv ../../2025-spring/exp02-sentiment-classificationn/dataset/test.csv
# 数据路径相对于运行时工作目录，输出目录必须为空或不存在。

"""Shared CSV and preprocessing contract for experiments 1 and 2 (stdlib only)."""
import csv
import hashlib
import html
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

PREPROCESSING = {
    "version": 1, "language": "English", "unicode": "NFKC",
    "lowercase": True, "html_unescape": True, "remove_html_tags": True,
    "token_pattern": r"[a-z]+(?:'[a-z]+)?|[0-9]+", "remove_stopwords": False,
    "combine": "title + space + text",
}


def tokenize(text):
    text = unicodedata.normalize("NFKC", html.unescape(text)).replace("’", "'")
    return re.findall(PREPROCESSING["token_pattern"], re.sub(r"<[^>]*>", " ", text).lower())


@dataclass(frozen=True)
class Review:
    label: int  # Amazon CSV: 1=negative, 2=positive. Lab 2 maps these to 0/1.
    title: str
    text: str

    def tokens(self):
        return tokenize(self.title + " " + self.text)

    def key(self):
        # Compare model-visible text, irrespective of sentiment label.
        return hashlib.sha256(" ".join(self.tokens()).encode()).digest()


def read_reviews(path):
    """Read headerless Amazon CSV or label/polarity,title,text/content header."""
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        first = True
        for row in reader:
            if not row or not any(s.strip() for s in row):
                continue
            fields = [s.strip().lower() for s in row]
            if first and fields in (["label", "title", "text"],
                                    ["polarity", "title", "text"],
                                    ["label", "title", "content"]):
                first = False
                continue
            first = False
            if len(row) != 3 or row[0].strip() not in {"1", "2"}:
                raise ValueError(f"{path}: CSV line {reader.line_num}: expected label 1/2,title,text")
            review = Review(int(row[0]), row[1].strip(), row[2].strip())
            if not review.tokens():
                raise ValueError(f"{path}: CSV line {reader.line_num}: empty English text")
            yield review


def write_reviews(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "title", "text"])
        writer.writerows((r.label, r.title, r.text) for r in rows)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def exclusion_keys(paths):
    return {row.key() for path in paths for row in read_reviews(path)}


def require_training_file(path):
    if Path(path).stem.lower() in {"dev", "test", "valid", "validation"}:
        raise ValueError("Do not use a development/test file for training. Supply a training CSV.")


def write_table(path, fields, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


# ===== 训练、分析与模型导出 =====
"""Run a controlled window=2 versus window=5 Word2Vec experiment."""
import argparse
from collections import Counter
import json
import importlib.metadata
import logging
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

QUERIES = ["good", "bad", "great", "phone", "battery", "screen", "price", "quality", "easy", "poor"]
PAIRS = [("good", "great"), ("good", "bad"), ("phone", "battery"), ("expensive", "cheap"), ("screen", "battery")]
FIXED_PARAMETERS = {"workers": 1, "shrink_windows": False, "alpha": 0.025,
                    "min_alpha": 0.0001, "negative": 5, "hs": 0, "sample": 1e-3}


def compare_neighbors(neighbors, queries):
    """比较两个窗口的近邻成员，不比较排名，也不据此判断模型优劣。"""
    comparison = []
    for query in dict.fromkeys(queries):
        # 复用已经计算好的近邻结果，保留各自的相似度排名顺序。
        groups = {window: [r for r in neighbors
                           if r["query"] == query and r["window"] == window]
                  for window in (2, 5)}
        left = [r["neighbor"] for r in groups[2] if r["status"] == "ok"]
        right = [r["neighbor"] for r in groups[5] if r["status"] == "ok"]
        comparable = all(groups[w] and all(r["status"] == "ok" for r in groups[w])
                         for w in (2, 5))
        # 集合用于判断成员是否相同；列表用于保持输出顺序稳定。
        common = [word for word in left if word in set(right)]
        comparison.append({
            "query": query,
            "window_2_neighbors": " | ".join(left),
            "window_5_neighbors": " | ".join(right),
            "common_neighbors": " | ".join(common) if comparable else "",
            "only_window_2": " | ".join(w for w in left if w not in set(right)) if comparable else "",
            "only_window_5": " | ".join(w for w in right if w not in set(left)) if comparable else "",
            "common_count": len(common) if comparable else "",
            "status": "ok" if comparable else "oov_or_missing",
        })
    return comparison


def export_handoff(model, folder, seed):
    import numpy as np
    words = ["<PAD>", "<UNK>"] + model.wv.index_to_key
    matrix = np.zeros((len(words), model.vector_size), dtype=np.float32)
    matrix[1] = np.random.default_rng(seed).normal(0, 0.05, model.vector_size)
    matrix[2:] = model.wv.vectors
    np.save(folder / "embedding.npy", matrix, allow_pickle=False)
    write_json(folder / "vocab.json", {"pad_id": 0, "unk_id": 1, "id_to_token": words,
               "token_to_id": {word: i for i, word in enumerate(words)}, "vector_size": model.vector_size})
    write_json(folder / "preprocessing.json", PREPROCESSING)


def project_vectors(model, words, method, seed, folder):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = model.wv[words]
    if method == "pca":
        from sklearn.decomposition import PCA
        xy = PCA(n_components=2, svd_solver="full").fit_transform(x)
    else:
        from sklearn.manifold import TSNE
        xy = TSNE(n_components=2, perplexity=min(30, max(1, (len(words)-1)/3)),
                  random_state=seed, init="pca", learning_rate="auto").fit_transform(x)
    write_table(folder / "projection.csv", ["word", "x", "y"],
                [{"word": w, "x": float(v[0]), "y": float(v[1])} for w, v in zip(words, xy)])
    fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
    ax.scatter(xy[:, 0], xy[:, 1], s=20, alpha=0.65, color="#285887")
    for word, (xx, yy) in zip(words[:30], xy[:30]):
        ax.annotate(word, (xx, yy), xytext=(3, 3), textcoords="offset points", fontsize=8)
    ax.set_title(f"{method.upper()} | window={model.window} | {len(words)} words (30 labelled)")
    ax.set_xlabel("Projection axis 1"); ax.set_ylabel("Projection axis 2")
    ax.text(0, -0.1, "Exploratory projection; axes of separately fitted models are not aligned.",
            transform=ax.transAxes, fontsize=9)
    fig.savefig(folder / f"{method}.png", dpi=160)
    plt.close(fig)


def experiment(a):
    import gensim
    import numpy as np
    from gensim.models import Word2Vec
    require_training_file(a.train_csv)
    rows = list(read_reviews(a.train_csv))
    if not rows:
        raise ValueError("Training CSV is empty")
    keys = [r.key() for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate training texts found; run prepare_data.py first")
    excluded = exclusion_keys(a.exclude_csv)
    overlap = len(set(keys) & excluded)
    if overlap:
        raise ValueError(f"{overlap} training texts also occur in development/test files")
    corpus = [r.tokens() for r in rows]
    counts = Counter(w for sentence in corpus for w in sentence)
    words = sorted((w for w in counts if counts[w] >= a.min_count), key=lambda w: (-counts[w], w))[:a.plot_words]
    if len(words) < 3:
        raise ValueError("Need at least 3 retained words. Increase training data or reduce --min-count")
    if a.output_dir.exists() and any(a.output_dir.iterdir()):
        raise ValueError("Output directory is not empty; choose a new run directory")
    a.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=a.output_dir / "training.log", level=logging.INFO,
                        format="%(asctime)s %(message)s", force=True)
    queries = QUERIES if a.query_file is None else [w.strip().lower() for w in a.query_file.read_text().splitlines() if w.strip()]
    write_json(a.output_dir / "run_manifest.json", {
        "status": "running", "training_name": a.train_csv.name, "training_sha256": sha256(a.train_csv),
        "rows": len(rows), "labels": dict(Counter(r.label for r in rows)),
        "token_count": sum(map(len, corpus)), "preprocessing": PREPROCESSING,
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items() if k != "exclude_csv"},
        "exclusion_files": [{"name": p.name, "sha256": sha256(p)} for p in a.exclude_csv],
        "overlap_count": overlap, "python": platform.python_version(), "gensim": gensim.__version__,
        "numpy": np.__version__, "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "packages": {p: importlib.metadata.version(p) for p in ['gensim','numpy','scipy','matplotlib','scikit-learn']},
        "fixed_parameters": FIXED_PARAMETERS,
        "label_encoding": {"1": "negative", "2": "positive"},
        "handoff": "window_5/embedding.npy + vocab.json + preprocessing.json",
    })
    results, pairs, neighbors = [], [], []
    for window in (2, 5):
        folder = a.output_dir / f"window_{window}"
        folder.mkdir()
        start = time.perf_counter()
        model = Word2Vec(vector_size=a.vector_size, window=window, min_count=a.min_count,
                         sg=int(a.algorithm == "skipgram"), seed=a.seed, **FIXED_PARAMETERS)
        model.build_vocab(corpus)
        model.train(corpus, total_examples=model.corpus_count, epochs=a.epochs, compute_loss=True)
        seconds = time.perf_counter() - start
        model.save(str(folder / "word2vec.model"))
        model.wv.save(str(folder / "vectors.kv"))
        export_handoff(model, folder, a.seed)
        project_vectors(model, words, a.projection, a.seed, folder)
        results.append({"window": window, "algorithm": a.algorithm, "vector_size": a.vector_size,
                        "epochs": a.epochs, "vocab_size": len(model.wv), "rows": len(rows),
                        "train_seconds": round(seconds, 4), "training_loss": model.get_latest_training_loss()})
        for left, right in PAIRS:
            ok = left in model.wv and right in model.wv
            pairs.append({"window": window, "word1": left, "word2": right,
                          "similarity": float(model.wv.similarity(left, right)) if ok else "",
                          "status": "ok" if ok else "oov"})
        for word in queries:
            if word not in model.wv:
                neighbors.append({"window": window, "query": word, "rank": "", "neighbor": "", "similarity": "", "status": "oov"})
                continue
            for rank, (near, score) in enumerate(model.wv.most_similar(word, topn=5), 1):
                neighbors.append({"window": window, "query": word, "rank": rank, "neighbor": near,
                                  "similarity": float(score), "status": "ok"})
        print(f"window={window}: {len(model.wv)} words, training {seconds:.2f}s", flush=True)
    write_table(a.output_dir / "comparison.csv", list(results[0]), results)
    write_table(a.output_dir / "word_pairs.csv", list(pairs[0]), pairs)
    write_table(a.output_dir / "neighbors.csv", ["window", "query", "rank", "neighbor", "similarity", "status"], neighbors)
    # 新增：直接汇总两个窗口的共同近邻和独有近邻，便于分析窗口的影响。
    neighbor_comparison = compare_neighbors(neighbors, queries)
    write_table(a.output_dir / "neighbor_comparison.csv",
                ["query", "window_2_neighbors", "window_5_neighbors", "common_neighbors",
                 "only_window_2", "only_window_5", "common_count", "status"],
                neighbor_comparison)
    manifest_path = a.output_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["status"] = "complete"
    write_json(manifest_path, manifest)
    print(f"Complete. Results and the lab-2 handoff are in {a.output_dir}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-csv", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("outputs/run1"))
    p.add_argument("--algorithm", choices=["cbow", "skipgram"], default="cbow")
    p.add_argument("--vector-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--min-count", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--plot-words", type=int, default=200)
    p.add_argument("--projection", choices=["pca", "tsne"], default="pca")
    p.add_argument("--query-file", type=Path)
    p.add_argument("--exclude-csv", type=Path, nargs="*", default=[])
    a = p.parse_args()
    if a.vector_size < 2 or a.epochs < 1 or a.min_count < 1 or a.plot_words < 3 or not 0 <= a.seed <= 2**32-1:
        p.error("Invalid dimension, epochs, min-count, plot-words, or seed")
    # Gensim initialization also depends on Python's process-level string hash.
    if os.environ.get("PYTHONHASHSEED") != str(a.seed):
        env = dict(os.environ, PYTHONHASHSEED=str(a.seed))
        return subprocess.call([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]], env=env)
    try:
        experiment(a)
    except (ValueError, OSError) as error:
        p.exit(1, f"Experiment failed: {error}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
