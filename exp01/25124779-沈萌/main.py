"""exp01:词向量表示与相似度分析

运行：python main.py --train-csv data/train_sample.csv
依赖：gensim numpy
"""
import argparse
import csv
import html
import os
import re
import subprocess
import sys
import unicodedata

# 0. 固定哈希种子
if os.environ.get("PYTHONHASHSEED") != "42":
    env = dict(os.environ, PYTHONHASHSEED="42")
    sys.exit(subprocess.call([sys.executable, os.path.abspath(__file__), *sys.argv[1:]], env=env))

import numpy as np
from gensim.models import Word2Vec

FIXED = dict(
    vector_size=128,
    min_count=2,
    sg=0,
    seed=42,
    workers=1,
    shrink_windows=False,
    alpha=0.025,
    min_alpha=0.0001,
    negative=5,
    hs=0,
    sample=1e-3,
)
EPOCHS = 10

# 1. 分词
TOKEN_PATTERN = re.compile(r"[a-z]+(?:'[a-z]+)?|[0-9]+")

def tokenize(text):
    text = unicodedata.normalize("NFKC", html.unescape(text)).replace("’", "'")
    text = re.sub(r"<[^>]*>", " ", text)
    return TOKEN_PATTERN.findall(text.lower())


def load_corpus(path):
    corpus, labels = [], []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            tokens = tokenize(row["title"] + " " + row["text"])
            if tokens:
                corpus.append(tokens)
                labels.append(row["label"])
    return corpus, labels

# 2. 训练两个模型
def train(corpus, window):
    model = Word2Vec(window=window, **FIXED)
    model.build_vocab(corpus)
    model.train(corpus, total_examples=model.corpus_count, epochs=EPOCHS)
    return model

# 3. 词对余弦
PAIRS = [("good", "great"), ("good", "bad"), ("phone", "battery"),
         ("expensive", "cheap"), ("screen", "battery")]
QUERIES = ["good", "price", "refund", "delivery"]

def report_pairs(models):
    print("\n【五组词对的余弦相似度】")
    print(f"{'词对':<22}{'window=2':>10}{'window=5':>10}")
    for a, b in PAIRS:
        row = f"{a + ' – ' + b:<22}"
        for w in (2, 5):
            kv = models[w].wv
            row += f"{kv.similarity(a, b):>10.3f}" if (a in kv and b in kv) else f"{'oov':>10}"
        print(row)

# 4. 近邻
def report_neighbors(models):
    print("\n【Top-5 近邻】")
    for word in QUERIES:
        print(f"  {word}")
        for w in (2, 5):
            kv = models[w].wv
            if word not in kv:
                print(f"    window={w}: OOV")
                continue
            top = ", ".join(f"{n}({s:.2f})" for n, s in kv.most_similar(word, topn=5))
            print(f"    window={w}: {top}")

# 主流程
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", default="data/train_sample.csv")
    args = parser.parse_args()

    if not os.path.exists(args.train_csv):
        sys.exit(f"找不到训练数据：{args.train_csv}")

    corpus, labels = load_corpus(args.train_csv)
    print(f"语料：{len(corpus)} 条评论，标签分布 {dict(zip(*np.unique(labels, return_counts=True)))}")

    models = {}
    for window in (2, 5):
        models[window] = train(corpus, window)
        print(f"window={window}: 词表 {len(models[window].wv)} 词")

    report_pairs(models)
    report_neighbors(models)
    print("\n完成。")

if __name__ == "__main__":
    main()
