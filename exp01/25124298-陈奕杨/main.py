"""实验 1：词向量表示与相似度分析 —— window=2 与 window=5 的受控对照。

除窗口外全部超参固定，两组的差异只能来自窗口大小。
用法：python main.py [训练CSV，缺省 data/train_sample.csv]
标注：[原] = 课程提供的实验代码（附 文件名:行号），逻辑未改；[改] = 单文件整合与输出改造
"""
import csv
import sys
import html
import re
import unicodedata
from pathlib import Path

from gensim.models import Word2Vec

# [原] lab_common.py:11-16 —— 预处理契约，两个窗口共用同一套，保证受控
PREPROCESSING = {
    "unicode": "NFKC", "lowercase": True, "html_unescape": True,
    "remove_html_tags": True, "token_pattern": r"[a-z]+(?:'[a-z]+)?|[0-9]+",
}
# [原] run_lab.py:16-17 —— 受控参数：两组完全一致，唯一变量是 window
FIXED_PARAMETERS = {"workers": 1, "shrink_windows": False, "alpha": 0.025,
                    "min_alpha": 0.0001, "negative": 5, "hs": 0, "sample": 1e-3}
# [原] run_lab.py:15 —— 5 组词对，覆盖同义、反义、部件、属性四类关系
PAIRS = [("good", "great"), ("good", "bad"), ("phone", "battery"),
         ("expensive", "cheap"), ("screen", "battery")]


# [原] lab_common.py:19-21，逐字未改
def tokenize(text):
    text = unicodedata.normalize("NFKC", html.unescape(text)).replace("’", "'")
    return re.findall(PREPROCESSING["token_pattern"], re.sub(r"<[^>]*>", " ", text).lower())


# [原] lab_common.py:38-58，仅去掉 Review 数据类，直接产出 token 列表
def read_reviews(path):
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
                raise ValueError(f"{path}: line {reader.line_num}: expected label 1/2,title,text")
            yield tokenize(row[1].strip() + " " + row[2].strip())


# [原] run_lab.py:100-103，逐字未改
def train(corpus, window):
    model = Word2Vec(vector_size=128, window=window, min_count=2, sg=0, seed=42, **FIXED_PARAMETERS)
    model.build_vocab(corpus)
    model.train(corpus, total_examples=model.corpus_count, epochs=10)
    return model


# [改] 原包 train() 之后把结果写进 CSV，这里改为打印终端对照表
def main(csv_path):
    corpus = list(read_reviews(csv_path))
    models = {}
    # [原] run_lab.py:98 —— 同一份语料跑两个窗口，唯一变化的是 window
    for window in (2, 5):
        models[window] = train(corpus, window)

    print(f"语料 {len(corpus)} 条 | 词表 {len(models[2].wv)} / {len(models[5].wv)} | 唯一变量 window")
    print(f"{'词对':<16}{'w=2':>8}{'w=5':>8}{'变化':>9}")
    # [原] run_lab.py:116 —— 遍历 5 组词对逐对算余弦
    for left, right in PAIRS:
        s2, s5 = models[2].wv.similarity(left, right), models[5].wv.similarity(left, right)
        print(f"{left + '-' + right:<18}{float(s2):>8.4f}{float(s5):>8.4f}{float(s5 - s2):>+9.4f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/train_sample.csv")
