"""实验 1｜词向量表示与相似度分析

学号：25124364    姓名：薛紫宸

运行：
    python main.py                       # 默认读 data/train_sample.csv
    python main.py --data 你的数据.csv    # 指定数据
    python main.py --demo                # 没有数据时，用内置迷你语料跑通流程

产出：
    控制台：窗口对比表 / 5 组词对 / 近邻 / 随机基线
    outputs/：embedding.npy、vocab.json、preprocessing.json、comparison.csv

依赖：gensim>=4.3, numpy
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import random
import re
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
from gensim.models import Word2Vec

# ----------------------------------------------------------------------------
# 配置：所有"受控参数"集中在这里，改参数只改这一处
# ----------------------------------------------------------------------------
SEED = 42
WINDOWS = (2, 5)                      # 唯一变量：上下文窗口
WORDS = ["good", "bad", "great", "cheap", "expensive", "phone", "battery", "screen"]
PAIRS = [("good", "great"), ("good", "bad"), ("phone", "battery"),
         ("expensive", "cheap"), ("screen", "battery")]

TRAIN_CFG = dict(
    vector_size=128,     # 向量维度
    min_count=2,         # 出现少于 2 次的词丢弃
    sg=0,                # 0 = CBOW（本实验固定）
    negative=5,          # 负采样个数
    hs=0,                # 不用层次 softmax
    sample=1e-3,         # 高频词下采样阈值
    alpha=0.025,         # 初始学习率
    min_alpha=0.0001,    # 终止学习率
    workers=1,           # 单线程：保证更新顺序可复现
    epochs=10,
    shrink_windows=False,  # 关掉随机窗口，否则窗口效应被随机性掩盖
    compute_loss=True,
)

DEMO = [s.split() for s in [
    "the camera is great but the battery life is short",
    "the camera lens is clear while the battery drains fast",
    "i love the camera and the battery is fine",
    "the screen is sharp but the battery is weak",
    "the screen is dull and the battery is strong",
    "this movie is good and funny",
    "this movie is bad and boring",
    "the soup was good and hot",
    "the soup was bad and cold",
    "i bought a new phone charger",
    "my phone battery is dead again",
    "the price is cheap and the service is good",
    "the price is expensive but the screen is great",
    "great product and cheap price",
]]


# ----------------------------------------------------------------------------
# 1) 读数据 + 分词
# ----------------------------------------------------------------------------
def tokenize(text: str) -> list[str]:
    """NFKC 规范化 → 反转义 HTML → 去标签 → 小写 → 切词。

    保留 not / no / never 等否定词：它们决定情感极性，删了标签会反向。
    数字也保留（屏幕尺寸、价格等常带数字）。
    """
    text = unicodedata.normalize("NFKC", html.unescape(str(text)))
    text = re.sub(r"<[^>]*>", " ", text)
    return re.findall(r"[a-z]+(?:'[a-z]+)?|[0-9]+", text.lower())


def load_corpus(path: Path) -> list[list[str]]:
    """CSV 每一行是一条评论：title + text 拼起来再分词。"""
    corpus: list[list[str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            text = f"{row.get('title', '')} {row.get('text', '')}".strip()
            tokens = tokenize(text)
            if tokens:
                corpus.append(tokens)
    return corpus


# ----------------------------------------------------------------------------
# 2) 训练
# ----------------------------------------------------------------------------
def train(corpus: list[list[str]], window: int) -> Word2Vec:
    return Word2Vec(corpus, window=window, seed=SEED, **TRAIN_CFG)


# ----------------------------------------------------------------------------
# 3) 分析：词对 / 近邻 / 随机基线
# ----------------------------------------------------------------------------
def pair_table(models: dict[int, Word2Vec]) -> dict[str, dict[int, float]]:
    table: dict[str, dict[int, float]] = {}
    for a, b in PAIRS:
        key = f"{a}-{b}"
        table[key] = {}
        for w, m in models.items():
            table[key][w] = float(m.wv.similarity(a, b)) if a in m.wv and b in m.wv else float("nan")
    return table


def random_baseline(model: Word2Vec, n_pairs: int = 20000) -> tuple[float, float]:
    """随机词对基线：从高频词里随机抽词对，看"随便两个词"能有多像。

    没有这个数，0.75 是高是低无从判断。
    """
    rng = random.Random(SEED)
    vocab = list(model.wv.index_to_key)[:300]
    sims = []
    for _ in range(n_pairs):
        a, b = rng.sample(vocab, 2)
        sims.append(float(model.wv.similarity(a, b)))
    arr = np.asarray(sims)
    return float(arr.mean()), float(np.percentile(arr, 95))


def top_neighbors(model: Word2Vec, topn: int = 5) -> dict[str, list[tuple[str, float]]]:
    out = {}
    for w in WORDS:
        if w in model.wv:
            out[w] = [(t, float(s)) for t, s in model.wv.most_similar(w, topn=topn)]
    return out


# ----------------------------------------------------------------------------
# 4) 导出交接文件（给实验 2 用）
# ----------------------------------------------------------------------------
def export_handoff(model: Word2Vec, outdir: Path) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    words = list(model.wv.index_to_key)
    mat = np.vstack([model.wv[w] for w in words]).astype(np.float32)

    # PAD=0 全零行；UNK=1 用固定种子生成的随机行（代表词表外词）
    unk = np.random.default_rng(SEED).normal(0, 0.1, size=mat.shape[1]).astype(np.float32)
    embedding = np.vstack([np.zeros((1, mat.shape[1]), np.float32), unk[None, :], mat])

    id_to_token = ["<pad>", "<unk>"] + words
    token_to_id = {t: i for i, t in enumerate(id_to_token)}

    np.save(outdir / "embedding.npy", embedding)
    with (outdir / "vocab.json").open("w", encoding="utf-8") as f:
        json.dump({"vector_size": int(mat.shape[1]),
                   "id_to_token": id_to_token,
                   "token_to_id": token_to_id}, f, ensure_ascii=False, indent=1)
    with (outdir / "preprocessing.json").open("w", encoding="utf-8") as f:
        json.dump({"lowercase": True, "keep_stopwords": True, "keep_negation": True,
                   "tokenizer": r"[a-z]+(?:'[a-z]+)?|[0-9]+",
                   "pad_id": 0, "unk_id": 1}, f, ensure_ascii=False, indent=1)

    # 自检：形状、类型、PAD 行、id 对齐
    assert embedding.dtype == np.float32
    assert embedding.shape == (len(words) + 2, int(mat.shape[1]))
    assert not embedding[0].any(), "PAD 行必须全零"
    assert np.linalg.norm(embedding[1]) > 0, "UNK 行不能是零向量"
    assert token_to_id["<pad>"] == 0 and token_to_id["<unk>"] == 1
    first = words[0]
    assert np.allclose(embedding[token_to_id[first]], model.wv[first]), "行号必须等于 token id"

    return {"vectors": int(embedding.shape[0]), "dim": int(embedding.shape[1])}


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------
def find_data() -> Path | None:
    for p in (Path("data/train_sample.csv"), Path("data/train.csv"),
              Path("../data/train_sample.csv"), Path("train_sample.csv")):
        if p.exists():
            return p
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=None)
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("outputs"))
    args = ap.parse_args()

    if not os.environ.get("PYTHONHASHSEED"):
        print("[提示] 未设置 PYTHONHASHSEED。本机通常仍可复现，跨机器请设 42。", flush=True)

    if args.demo:
        corpus, source = DEMO, "内置迷你语料（--demo）"
        cfg = dict(TRAIN_CFG, vector_size=32, min_count=1, epochs=80)
        print("[注意] --demo 只为验证流程能跑通；14 句语料上的数值没有分析意义"
              "（小语料会坍缩，见练习册题 3）。正式结果请用 --data 指定真实数据。")
    else:
        path = args.data or find_data()
        if path is None:
            print("找不到数据文件。请用 --data 指定，或先用 --demo 跑通流程。", file=sys.stderr)
            return 2
        corpus = load_corpus(path)
        source, cfg = str(path), TRAIN_CFG

    n_tokens = sum(len(s) for s in corpus)
    n_types = len({w for s in corpus for w in s})
    print(f"数据：{source}")
    print(f"句子数 {len(corpus)} | token 总数 {n_tokens:,} | 不同词 {n_types:,}")

    models: dict[int, Word2Vec] = {}
    for w in WINDOWS:
        t0 = time.perf_counter()
        m = Word2Vec(corpus, window=w, seed=SEED, **cfg)
        dt = time.perf_counter() - t0
        loss = float(getattr(m, "running_training_loss", float("nan")))
        models[w] = m
        print(f"window={w} | 词表 {len(m.wv):,} | 耗时 {dt:.3f}s | 累计 loss {loss:,.1f}")

    print("\n== 5 组词对的余弦相似度 ==")
    table = pair_table(models)
    for key, row in table.items():
        cells = "  ".join(f"w{w}={row[w]:.3f}" for w in WINDOWS)
        print(f"{key:22s} {cells}")

    print("\n== 随机词对基线（衡量“0.75 算不算高”）==")
    for w, m in models.items():
        mean, p95 = random_baseline(m)
        print(f"window={w} | 均值 {mean:.3f} | 95% 分位 {p95:.3f}")

    print("\n== Top-5 近邻 ==")
    for w, m in models.items():
        print(f"-- window={w}")
        for word, nbrs in top_neighbors(m).items():
            print(f"   {word:10s} " + ", ".join(f"{t}({s:.2f})" for t, s in nbrs))

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    with (out / "comparison.csv").open("w", encoding="utf-8", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["pair"] + [f"window_{w}" for w in WINDOWS])
        for key, row in table.items():
            wr.writerow([key] + [f"{row[w]:.4f}" for w in WINDOWS])

    last = models[max(WINDOWS)]
    info = export_handoff(last, out)
    print(f"\n交接文件已写入 {out}：embedding.npy {info['vectors']}×{info['dim']}，vocab.json，preprocessing.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
