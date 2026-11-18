import logging
import time
from pathlib import Path
from collections import Counter

from lab_common import (
    PREPROCESSING,
    tokenize,
    read_reviews,
    exclusion_keys,
    require_training_file,
    write_json,
    write_table,
)
print(PREPROCESSING)
train_csv = Path("data/train_sample.csv")
rows = list(read_reviews(train_csv))

print(len(rows))

corpus = [r.tokens() for r in rows]#在read_reviews已经把review转换为Review类
counts = Counter(word for sentence in corpus for word in sentence)

print(len(rows), sum(map(len, corpus)), len(counts),corpus[0])

seen = set()
unique_rows = []

for row in rows:
    key = row.key()
    if key in seen:
        continue
    seen.add(key)
    unique_rows.append(row)

rows = unique_rows#消除row中重复评论

corpus = [r.tokens() for r in rows]#拼接title和text
counts = Counter(word for sentence in corpus for word in sentence)

print(len(rows), sum(map(len, corpus)), len(counts))
require_training_file(train_csv)

exclude_csv = [
    Path("../../2025-spring/exp02-sentiment-classificationn/dataset/dev.csv"),
    Path("../../2025-spring/exp02-sentiment-classificationn/dataset/test.csv"),
]

keys = [r.key() for r in rows]

if len(set(keys)) != len(keys):
    raise ValueError("训练文本存在重复，需要先处理重复数据")

excluded = exclusion_keys(exclude_csv)
overlap = len(set(keys) & excluded)

if overlap:
    raise ValueError(f"训练文本与开发集或测试集有 {overlap} 条重叠")

print("训练文本无重复；与开发集、测试集重叠数：", overlap)
print(len(keys), len(set(keys)))
output_dir = Path("outputs/notebook_run1")
output_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=output_dir / "training.log",
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    encoding="utf-8",
    force=True
)

logging.info("实验开始")
logging.info("预处理规则：%s", PREPROCESSING)
logging.info(
    "去重后评论数=%d，Token总数=%d，低频过滤前词表大小=%d",
    len(rows), sum(map(len, corpus)), len(counts)
)
logging.info("训练文本无重复，与开发集、测试集重叠数=%d", overlap)

from gensim.models import Word2Vec
from run_lab import FIXED_PARAMETERS

vector_size = 128
epochs = 10
min_count = 2
seed = 42

logging.info(
    "算法=CBOW，维度=%d，轮数=%d，min_count=%d，seed=%d，固定参数=%s",
    vector_size, epochs, min_count, seed, FIXED_PARAMETERS
)

models = {}
results = []

for window in (2, 5):
    logging.info("开始训练 window=%d", window)
    start = time.perf_counter()

    model = Word2Vec(
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=0,
        seed=seed,
        **FIXED_PARAMETERS
    )

    model.build_vocab(corpus)

    model.train(
        corpus,
        total_examples=model.corpus_count,
        epochs=epochs,
        compute_loss=True
    )

    seconds = time.perf_counter() - start

    models[window] = model

    results.append({
        "window": window,
        "algorithm": "cbow",
        "vector_size": vector_size,
        "epochs": epochs,
        "vocab_size": len(model.wv),
        "rows": len(rows),
        "train_seconds": round(seconds, 4),
        "training_loss": model.get_latest_training_loss()
    })

    logging.info("训练完成：%s", results[-1])
    print(
        f"window={window}，词表大小={len(model.wv)}，"
        f"训练耗时={seconds:.2f}秒"
    )

write_table(
    output_dir / "comparison.csv",
    list(results[0]),
    results
)

print(results)
from run_lab import PAIRS

pairs = []

for window in (2,5):
    model = models[window]

    for left,right in PAIRS:
        ok = left in model.wv and right in model.wv

        pairs.append ({
            "window":window,
            "word1":left,
            "word2":right,
            "similarity":float(model.wv.similarity(left,right)) if ok else "",
            "status":"ok" if ok else "OOV"

        })

write_table(
    output_dir / "word_pairs.csv",["window","word1","word2","similarity","status"],pairs

)
print(pairs)

QUERIES = ["good", "bad", "great", "phone", "battery", "screen", "price", "quality", "easy", "poor"]

neighbors = []

for window in (2,5):
    model = models[window]
    for word in QUERIES:
        if word not in model.wv:
            neighbors.append({
            "window" : window,
            "query" : word,
            "rank":"",
            "neighbor":"",
            "similarity":"",
            "status":"OOV"
            })
            continue
        for rank,(near,score) in enumerate(model.wv.most_similar(word,topn=5),1):
            neighbors.append({
                "window": window,
                "query":word,
                "rank":rank,
                "neighbor":near,
                "similarity": float(score),
                "status":"ok"
            })
write_table(
    output_dir / "neighbors.csv",
    ["window", "query", "rank", "neighbor", "similarity", "status"],
    neighbors
)

print(neighbors)
from run_lab import project_vectors

words=sorted(
    (word for word in counts if counts[word]>=min_count),key = lambda word:(-counts[word],word)
)[:200]

for window in (2,5):
    folder = output_dir / f"window_{window}"
    folder.mkdir(parents=True,exist_ok=True)

    project_vectors(
        models[window],
        words,
        "pca",
        seed,
        folder
    )
from run_lab import export_handoff

for window in (2,5):
    model = models[window]
    folder = output_dir/f"window_{window}"
    folder.mkdir(parents=True,exist_ok=True)

    model.save(str(folder/"word2vec.model"))
    model.wv.save(str(folder/"vectors.kv"))

    export_handoff(model, folder, seed)
import json
import numpy as np

folder = output_dir / "window_5"

matrix = np.load(folder/"embedding.npy",allow_pickle = False)

with (folder / "vocab.json").open(encoding="utf-8") as f:
    vocab = json.load(f)

assert matrix.shape ==(
    len(vocab["id_to_token"]),vocab["vector_size"]
)

ids = [
    vocab["token_to_id"].get(word, vocab["unk_id"])
    for word in tokenize("not a good phone")
]

print(matrix.shape, ids)
from gensim.models import KeyedVectors#测试现场查询

vectors = KeyedVectors.load(
    str(output_dir / "window_5" / "vectors.kv")
)

query = "good"
tokens = tokenize(query)

if len(tokens) == 1 and tokens[0] in vectors:
    print(vectors.most_similar(tokens[0], topn=5))
else:
    print("OOV 或处理后不是单个词，无法查询")