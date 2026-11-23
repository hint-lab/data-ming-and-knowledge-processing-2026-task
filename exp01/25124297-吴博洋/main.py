"""比较两个窗口的近邻，统计查询词和近邻的语料覆盖。"""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
LAB_DIR = HERE.parent
sys.path.insert(0, str(LAB_DIR))
from lab_common import PREPROCESSING, read_reviews, sha256, tokenize

DEFAULT_WORDS = ["phone", "sound", "service"]
TOPN = 5


def load_models(model_dir, train_csv):
    """读取训练记录和两组词向量。"""
    manifest = json.loads((model_dir / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError("模型训练尚未完成。")
    if manifest.get("training_sha256") != sha256(train_csv):
        raise ValueError("训练CSV的文件摘要与训练记录不一致。")
    if manifest.get("preprocessing") != PREPROCESSING:
        raise ValueError("分词规则与训练记录不一致。")

    try:
        from gensim.models import KeyedVectors
    except ImportError as error:
        raise ValueError("当前Python环境缺少模型依赖。") from error
    models = {}
    for window in (2, 5):
        path = model_dir / f"window_{window}" / "vectors.kv"
        models[window] = KeyedVectors.load(str(path))
    return models, manifest


def find_neighbors(models, words):
    """为每个查询词找两组近邻，同时收集需要统计词频的全部词。"""
    results = {}
    targets = set(words)
    for word in words:
        results[word] = {}
        for window, vectors in models.items():
            if word in vectors:
                topn = min(TOPN, len(vectors) - 1)
                neighbors = vectors.most_similar(word, topn=topn)
            else:
                neighbors = None  # OOV：词表外。
            results[word][window] = neighbors
            if neighbors is not None:
                targets.update(near for near, score in neighbors)
    return results, targets


def count_coverage(train_csv, targets):
    """统计出现次数与评论条数：同一条评论内的重复只给评论数加1。"""
    token_counts = Counter()
    review_counts = Counter()
    total_reviews = 0
    for review in read_reviews(train_csv):
        tokens = review.tokens()
        token_counts.update(word for word in tokens if word in targets)
        review_counts.update(word for word in set(tokens) if word in targets)
        total_reviews += 1
    return token_counts, review_counts, total_reviews


def make_report(results, token_counts, review_counts, total_reviews, manifest):
    """将查询结果和语料统计排成Markdown表格。"""
    lines = ["# 双窗口词向量观察结果", "",
             f"分析范围：{total_reviews} 条训练评论；每组最多 {TOPN} 个近邻。",
             f"训练数据 SHA256：`{manifest['training_sha256']}`。", ""]
    for word, windows in results.items():
        lines.extend([f"## {word}", "",
                      f"查询词出现 **{token_counts[word]} 次**，覆盖 **{review_counts[word]} 条评论**。", ""])
        for window, neighbors in windows.items():
            lines.extend([f"### window={window}", ""])
            if neighbors is None:
                lines.extend(["OOV（词表外）。", ""])
                continue
            lines.extend(["| 排名 | 近邻 | 余弦相似度 | 出现次数 | 评论条数 |",
                          "|---:|---|---:|---:|---:|"])
            for rank, (near, score) in enumerate(neighbors, 1):
                lines.append(f"| {rank} | {near} | {score:.6f} | {token_counts[near]} | {review_counts[near]} |")
            lines.append("")

        if any(neighbors is None for neighbors in windows.values()):
            lines.extend(["近邻集合重合度：—。", ""])
            continue
        first = {near for near, score in windows[2]}
        second = {near for near, score in windows[5]}
        common = first & second  # &取两个集合中都有的词。
        union = first | second   # |合并两组词，相同的词只保留一次。
        lines.extend(["共同近邻：" + ("、".join(sorted(common)) if common else "无") + "。", ""])
        if union:
            overlap = len(common) / len(union)
            lines.extend([f"近邻集合重合度（Jaccard）= {len(common)} ÷ {len(union)} = **{overlap:.2%}**。", ""])
        else:
            lines.extend(["近邻集合重合度：—。", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("words", nargs="*", help="用空格分隔英文词；不填则在运行时询问")
    parser.add_argument("--model-dir", type=Path, default=LAB_DIR / "outputs" / "formal")
    parser.add_argument("--train-csv", type=Path, default=LAB_DIR / "data" / "train_sample.csv")
    parser.add_argument("--output", type=Path, default=HERE / "观察结果.md")
    args = parser.parse_args()
    try:
        inputs = args.words
        if not inputs:
            text = input("请输入英文词，用空格分开（直接回车观察 phone sound service）：\n> ")
            inputs = text.split() or DEFAULT_WORDS
        words = []
        for text in inputs:
            tokens = tokenize(text)
            if len(tokens) != 1:
                raise ValueError(f"查询词格式错误：{text!r}。")
            if tokens[0] not in words:
                words.append(tokens[0])

        print("正在读取已有模型和训练语料……", flush=True)
        models, manifest = load_models(args.model_dir, args.train_csv)
        results, targets = find_neighbors(models, words)
        token_counts, review_counts, total = count_coverage(args.train_csv, targets)
        report = make_report(results, token_counts, review_counts, total, manifest)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
        print(report)
        print(f"报告已保存到：{args.output.resolve()}")
    except (OSError, ValueError, EOFError) as error:
        parser.exit(1, f"运行失败：{error}\n")


if __name__ == "__main__":
    # 让Windows终端正确显示中文。
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    main()
