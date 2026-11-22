import argparse
import csv
import hashlib
import html
import json
import random
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
from gensim.models import Word2Vec

TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")

# 否定词不参与停用词过滤，否则 good/bad 的语境会被抹平
NEGATIONS = {
    "not", "no", "never", "nor", "none", "cannot", "n't",
    "don't", "doesn't", "didn't", "isn't", "wasn't", "aren't", "weren't",
    "can't", "couldn't", "won't", "wouldn't", "shouldn't",
    "haven't", "hasn't", "hadn't", "ain't", "mustn't", "needn't",
}

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "than", "so",
    "of", "to", "in", "on", "at", "by", "for", "with", "about", "from",
    "as", "into", "over", "after", "before", "between", "during",
    "is", "am", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "doing", "done", "have", "has", "had", "having",
    "i", "me", "my", "we", "our", "you", "your", "he", "she", "it", "its",
    "they", "them", "their", "this", "that", "these", "those",
    "there", "here", "when", "where", "which", "who", "whom", "what", "why",
    "will", "would", "shall", "should", "can", "could", "may", "might", "must",
    "very", "too", "just", "also", "only", "really", "quite", "much", "many",
    "some", "any", "all", "each", "every", "both", "either", "neither",
    "s", "t", "re", "ve", "ll", "d", "m",
}

SHORT_KEEP = {"a", "i", "no", "ok", "tv", "cd", "am", "pm", "mr", "ms"}


def clean_text(text):
    s = html.unescape(str(text or ""))
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"https?://\S+|www\.\S+", " ", s)
    return s.replace("’", "'").replace("‘", "'")


def tokenize(text):
    tokens = TOKEN_RE.findall(clean_text(text).lower())
    out = []
    for tok in tokens:
        if tok in NEGATIONS:
            out.append(tok)
        elif tok in STOPWORDS:
            continue
        elif len(tok) < 2 and tok not in SHORT_KEEP:
            continue
        else:
            out.append(tok)
    return out


def fingerprint(text):
    normalized = re.sub(r"\s+", " ", str(text or "").lower()).strip()
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def read_reviews(path):
    rows = []
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
        for i, raw in enumerate(csv.reader(f)):
            if not raw or all(not c.strip() for c in raw):
                continue
            if i == 0 and not raw[0].strip().isdigit():
                continue
            if len(raw) < 3:
                continue
            label = int(raw[0].strip())
            if label not in (1, 2):
                raise ValueError(f"{path}: 标签必须是 1 或 2，实际为 {label}")
            rows.append({"label": label, "title": raw[1], "text": raw[2]})
    if not rows:
        raise ValueError(f"{path}: 没有解析出任何评论")
    return rows


def load_exclusions(paths):
    keys = set()
    for path in paths:
        for row in read_reviews(path):
            keys.add(fingerprint(row["title"] + " " + row["text"]))
            keys.add(fingerprint(row["text"]))
    return keys


def check_isolation(rows, excluded):
    seen = set()
    duplicates = 0
    overlap = 0
    for row in rows:
        key = fingerprint(row["title"] + " " + row["text"])
        if key in seen:
            duplicates += 1
        seen.add(key)
        if key in excluded:
            overlap += 1
    if duplicates:
        raise ValueError(f"训练集内部有 {duplicates} 条重复，请先去重")
    if overlap:  # 有重叠直接报错，不做静默剔除
        raise ValueError(f"{overlap} 条训练文本同时出现在 dev/test 中，存在数据泄漏")


def build_corpus(rows, min_count):
    sentences = []
    for row in rows:
        tokens = tokenize(f"{row['title']}. {row['text']}")
        if tokens:
            sentences.append(tokens)

    counter = Counter(tok for sentence in sentences for tok in sentence)
    kept = {w for w, c in counter.items() if c >= min_count}

    stats = {
        "rows": len(sentences),
        "tokens": sum(counter.values()),
        "types": len(counter),
        "vocab": len(kept),
    }
    return sentences, stats


def train_one(sentences, window, cfg, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    model = Word2Vec(
        vector_size=cfg["vector_size"],
        window=window,
        min_count=cfg["min_count"],
        sg=0,
        workers=1,
        seed=cfg["seed"],
        epochs=1,
        shrink_windows=False,  # 关闭随机窗口，2/5 才是字面距离
        compute_loss=True,
        sorted_vocab=1,
    )
    model.build_vocab(sentences)

    losses = []
    start = time.perf_counter()
    for _ in range(cfg["epochs"]):
        model.running_training_loss = 0.0  # 每轮清零，拿到的是单轮 loss
        model.train(
            sentences,
            total_examples=model.corpus_count,
            epochs=1,
            compute_loss=True,
        )
        losses.append(float(model.get_latest_training_loss()))
    seconds = time.perf_counter() - start

    model.save(str(out_dir / "word2vec.model"))
    model.wv.save(str(out_dir / "vectors.kv"))

    return {
        "window": window,
        "vocab": len(model.wv),
        "seconds": round(seconds, 2),
        "loss": round(losses[-1], 2),
    }


def word_pairs(models, pairs):
    print("\n词对余弦相似度")
    print(f"{'词对':<22}{'window=2':>10}{'window=5':>10}")
    for left, right in pairs:
        values = []
        for window in sorted(models):
            wv = models[window].wv
            if left in wv and right in wv:
                values.append(f"{wv.similarity(left, right):.4f}")
            else:
                values.append("oov")
        print(f"{left + '-' + right:<22}{values[0]:>10}{values[1]:>10}")


def top_neighbors(models, queries, k=5):
    for word in queries:
        print(f"\n[{word}]")
        for window in sorted(models):
            wv = models[window].wv
            if word not in wv:
                print(f"  window={window}: OOV")
                continue
            neighbors = wv.most_similar(word, topn=k)
            print(f"  window={window}: " + " | ".join(f"{w}({s:.3f})" for w, s in neighbors))


def export_handoff(model, out_dir, seed):
    out_dir.mkdir(parents=True, exist_ok=True)
    tokens = list(model.wv.index_to_key)
    dim = model.vector_size

    matrix = np.zeros((len(tokens) + 2, dim), dtype=np.float32)  # 第 0 行 PAD 全零
    matrix[2:] = model.wv.vectors.astype(np.float32)

    rng = np.random.default_rng(seed)
    unk = rng.normal(size=dim).astype(np.float32)
    unk /= np.linalg.norm(unk) + 1e-9
    matrix[1] = unk * np.linalg.norm(matrix[2:], axis=1).mean()

    assert matrix.shape == (len(tokens) + 2, dim)
    assert (matrix[0] == 0).all()

    np.save(out_dir / "embedding.npy", matrix, allow_pickle=False)

    vocab = {
        "pad_id": 0,
        "unk_id": 1,
        "id_to_token": ["<pad>", "<unk>"] + tokens,
        "token_to_id": {tok: i + 2 for i, tok in enumerate(tokens)},
        "vector_size": dim,
    }
    with open(out_dir / "vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="实验1 核心算法")
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--exclude-csv", type=Path, nargs="*", default=[])
    parser.add_argument("--out", type=Path, default=Path("outputs"))
    parser.add_argument("--vector-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--min-count", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    cfg = {
        "vector_size": args.vector_size,
        "epochs": args.epochs,
        "min_count": args.min_count,
        "seed": args.seed,
    }

    print("[1/5] 读取训练数据")
    rows = read_reviews(args.train_csv)
    print(f"      {len(rows)} 条评论")

    print("[2/5] 检查数据隔离")
    excluded = load_exclusions(args.exclude_csv)
    check_isolation(rows, excluded)
    print("      与 dev/test 无重叠")

    print("[3/5] 分词与词频统计")
    sentences, stats = build_corpus(rows, args.min_count)
    print(f"      有效 token {stats['tokens']}，词表 {stats['types']} → {stats['vocab']}")

    print("[4/5] 训练两组窗口")
    models = {}
    for window in (2, 5):
        out_dir = args.out / f"window_{window}"
        info = train_one(sentences, window, cfg, out_dir)
        models[window] = Word2Vec.load(str(out_dir / "word2vec.model"))
        print(f"      window={window}: 词表 {info['vocab']}，耗时 {info['seconds']}s，"
              f"loss {info['loss']:,.0f}")

    print("[5/5] 实测与导出")
    word_pairs(models, [
        ("good", "great"), ("good", "bad"), ("phone", "battery"),
        ("hot", "cold"), ("movie", "film"),
    ])
    top_neighbors(models, [
        "phone", "battery", "screen", "good", "movie",
        "kindle", "charger", "refund", "love", "flimsy",
    ])
    export_handoff(models[5], args.out / "window_5", args.seed)
    print(f"\n完成：结果在 {args.out}")


if __name__ == "__main__":
    main()
