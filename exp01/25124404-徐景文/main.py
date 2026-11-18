# 运行环境：Windows PowerShell，已 pip install gensim pandas numpy
# ============================================================
import html
import unicodedata
import csv      # 读 csv 文件（类比 C 里的 fopen + fgets，但更省事）
import json     # 读写 json 文件（词表要用它存）
import os       # 建目录、拼路径
import re       # 正则表达式，用来切词
import time     # 计时，报告里要写两个窗口各自的训练耗时
from collections import Counter          # 数词频神器：Counter(["a","b","a"]) -> {"a":2,"b":1}
import numpy as np                       # 数值数组（类比 C 的 float 数组，但不用自己管内存）
from gensim.models import Word2Vec       # 训练词向量的核心库

SEED = 42   # 随机种子：固定它，别人重跑能得到和我一样的向量（可复现性）


# ============================================================
# 第 1 步：分词（tokenize）
# 老师数据是英文评论，目标是把一条长字符串切成小写单词列表
# ============================================================
def tokenize(text):
    text = unicodedata.normalize("NFKC", html.unescape(text)).replace("’", "'")
    text = re.sub(r"<[^>]*>", " ", text).lower()
    # 统一转小写，"Good" 和 "good" 算同一个词
    # TODO 1：用 re.findall 把 text 切成词列表并返回
    # 提示（老师课件同款思路，绕开 nltk 下载问题）：
    #   模式 r"[a-z]+(?:'[a-z]+)?|[0-9]+"
    #   含义：连续小写字母（允许中间带一个 ' 如 don't），或者连续数字
    # 一行即可：return re.findall(r"...", text)
    return re.findall(r"[a-z]+(?:'[a-z]+)?|[0-9]+", text)


# ============================================================
# 第 2 步：读语料（csv -> 句子列表）
# csv 每行是一条评论，三列：label, title, text
# 我们要的是 text 列（评论文本），把它 tokenize 成一个词列表
# ============================================================
def load_corpus(csv_path):
    sentences = []  # 形如 [ ["i","bought","this","cd",...], [...], ... ]
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)  # 按列名读，每行是一个字典 {"label":..,"title":..,"text":..}
        for row in reader:
            tokens = tokenize(row["title"] + " " + row["text"])
            if len(tokens) > 0:
                sentences.append(tokens)
    return sentences


# ============================================================
# 第 3 步：词频统计 + 过滤低频词
# 为什么过滤：只出现 1~2 次的词学不出有意义的向量（长尾噪声），
# min_count 参数就是干这个的——但先自己数一遍才能写进报告
# ============================================================
def word_stats(sentences, top_n=10):
    # TODO 2：统计整个语料的词频
    # 提示：
    #   1) 先把所有句子的词摊平成一个总列表
    #      （可以用双重循环，也可以一句 all_tokens.extend(s)）
    #   2) 用 Counter(all_tokens) 得到词频字典
    #   3) 返回 Counter 对象
    all_tokens = []
    for s in sentences:
        all_tokens.extend(s)
    return Counter(all_tokens)


# ============================================================
# 第 4 步：训练 Word2Vec
# 对比实验核心：同一个语料，分别用 window=2 和 window=5 训练两次
# window=2  -> 小窗口，学"近距离搭配"（语法/短语层面）
# window=5  -> 大窗口，学"远距离主题"（语义/话题层面）
# 其他参数照老师课件（受控实验：除窗口外全部固定！）
# ============================================================
def train_model(sentences, window, sg=0, vector_size=128, min_count=2, epochs=10):
    t0 = time.time()  # 记下开始时间
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,   # 每个词用 128 维稠密向量表示
        window=window,             # ← 本实验唯一变化的变量
        min_count=min_count,       # 出现少于 2 次的词直接不进词表
        sg=sg,                     # 0 = CBOW（老师默认）；拓展实验才改 1 = skip-gram
        epochs=epochs,             # 语料过 10 遍（老师默认 10，别用库默认的 5）
        workers=1,                 # 单线程，保证可复现
        seed=SEED,
        negative=5,                # 每个正样本配 5 个负样本（负采样加速）
        hs=0,                      # 不用层次 softmax，用负采样
        sample=1e-3,               # 高频词（如 the）下采样阈值
        alpha=0.025,               # 初始学习率
        min_alpha=0.0001,          # 最终学习率（训练中线性衰减）
        shrink_windows=False,      # 关闭"动态缩小窗口"，让两种设定差别容易解释
    )
    elapsed = time.time() - t0
    print(f"window={window} 训练完成，耗时 {elapsed:.1f} 秒，词表 {len(model.wv)} 词")
    return model, elapsed


# ============================================================
# 第 5 步：相似度对照表（报告第 2 项交付物的数据来源）
# 5 组固定词对，两个窗口各测一遍余弦相似度
# ============================================================
PAIRS = [
    ("good", "great"),        # 近义词：直觉上应该相似度高
    ("good", "bad"),          # 反义词：直觉上应该低，实测反而高 -> 异常案例1
    ("phone", "battery"),     # 同主题词
    ("expensive", "cheap"),   # 反义词
    ("screen", "battery"),    # 同主题词
]

def similarity_report(model2, model5):
    print(f"{'词对':<22}{'window=2':>10}{'window=5':>10}")
    for w1, w2 in PAIRS:
        # TODO 3：对每个词对，分别在两个模型上算余弦相似度
        # 提示：model.wv.similarity(w1, w2) 返回 -1~1 的浮点数
        #       注意用 try/except 包一下，防止某个词不在词表里（KeyError）
        try:
            s2 = model2.wv.similarity(w1, w2)
            s5 = model5.wv.similarity(w1, w2)
        except KeyError:
            print(f"{w1 + '-' + w2:<22}{'OOV':>10}{'OOV':>10}")
            continue
        print(f"{w1 + '-' + w2:<22}{s2:>10.4f}{s5:>10.4f}")



# ============================================================
# 第 6 步：近邻查询（异常分析 + 现场验收要用的）
# 老师验收可能现场给一个新词，让你当场跑 most_similar 并解释
# ============================================================
def nearest(model, word, topn=5):
    # TODO 4：返回与 word 最相近的 topn 个词
    # 提示：model.wv.most_similar(word, topn=topn)
    #       返回 [(词, 相似度), ...] 的列表
    return model.wv.most_similar(word, topn=topn)



# ============================================================
# 第 7 步：导出交付物（评分标准"成果交接完整性 10 分"）
# 实验2 要直接 np.load 这个矩阵，所以行号规则必须和老师约定一致：
#   第 0 行 = <PAD>（全 0，填充用）
#   第 1 行 = <UNK>（固定随机向量，遇到词表外的词就用它）
#   第 2 行起 = 真实词向量，按词频从高到低排列
# 为什么不能直接用 model.wv.vectors：它没有 PAD/UNK 两行，
# 实验2 按"词ID=行号"查表时会整体错位两行，向量全乱。
# ============================================================
def export_artifacts(model, out_dir, window):
    words = model.wv.index_to_key          # 按词频降序排好的词表
    dim = model.vector_size

    # 行 0 默认就是全 0（np.zeros），正好当 PAD
    matrix = np.zeros((len(words) + 2, dim), dtype=np.float32)
    # 行 1 = UNK：用固定种子生成，保证每次导出都一样（可复现）
    matrix[1] = np.random.default_rng(SEED).normal(0, 0.05, dim)
    # 行 2 起：把模型学到的向量整块搬进来（形状 (词数, 128) 直接对位）
    matrix[2:] = model.wv.vectors

    # 双向词表：token_to_id 给"词→ID"，id_to_token 给"ID→词"（调试时用）
    vocab = {
        "token_to_id": {w: i + 2 for i, w in enumerate(words)},
        "id_to_token": {str(i + 2): w for i, w in enumerate(words)},
        "vector_size": dim,
        "pad_id": 0,
        "unk_id": 1,
    }

    # 预处理约定：实验2 必须用完全相同的分词规则，否则同一个词对应错行
    preprocessing = {
        "lowercase": True,
        "token_pattern": r"[a-z]+(?:'[a-z]+)?|[0-9]+",
        "html_unescape": True,
        "strip_html_tags": True,
        "min_count": 2,
        "window": window,
        "vector_size": dim,
        "algorithm": "cbow",
        "epochs": 10,
        "seed": SEED,
    }

    os.makedirs(out_dir, exist_ok=True)   # 目录不存在就建（类比 C 的 mkdir，但不会因已存在而报错）
    np.save(os.path.join(out_dir, f"embedding_w{window}.npy"), matrix)
    with open(os.path.join(out_dir, f"vocab_w{window}.json"), "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, f"preprocessing_w{window}.json"), "w", encoding="utf-8") as f:
        json.dump(preprocessing, f, ensure_ascii=False, indent=2)
    print(f"已导出 window={window}：矩阵形状 {matrix.shape}，词表 {len(words)} 词")


# ============================================================
# 主流程：像 C 语言的 main()
# ============================================================
if __name__ == "__main__":
    CSV_PATH = "data/train_sample.csv"   # 相对路径：在你自己的实验文件夹下运行
    OUT_DIR = "outputs/my_exp01"

    print("== 第1步：读语料并分词 ==")
    sentences = load_corpus(CSV_PATH)
    print(f"共 {len(sentences)} 条评论")

    print("== 第2步：词频统计 ==")
    counter = word_stats(sentences)
    print(f"词表总大小（未过滤）：{len(counter)}")
    print(f"最高频 10 个词：{counter.most_common(10)}")

    print("== 第3步：训练 window=2 ==")
    model2, t2 = train_model(sentences, window=2)

    print("== 第4步：训练 window=5 ==")
    model5, t5 = train_model(sentences, window=5)

    print("== 第5步：相似度对照表 ==")
    similarity_report(model2, model5)

    print("== 第6步：关键词近邻（异常分析用）==")
    for word in ["good", "bad", "great", "screen", "battery"]:
        print(f"--- {word} ---")
        print("window=2:", nearest(model2, word))
        print("window=5:", nearest(model5, word))
    print("关键词语料频次：")
    for word in ["good", "bad", "great", "screen", "battery", "cheap", "expensive"]:
        print(f"  {word}: {counter[word]} 次")

    print("== 第6.5步：低频词共现检查（案例2素材）==")
    for target in ["screen", "battery", "phone"]:
        co = Counter()
        docs = 0
        for s in sentences:
            if target in s:
                docs += 1
                co.update(w for w in s if w != target)
        print(f"{target}: 出现在 {docs} 条评论中，最常共现的词：{co.most_common(8)}")

    print("== 第6.6步：排除高频功能词后的共现 ==")
    stopwords = {w for w, _ in counter.most_common(30)}
    for target in ["screen", "battery", "phone"]:
        co = Counter()
        for s in sentences:
            if target in s:
                co.update(w for w in s if w != target and w not in stopwords)
        print(f"{target}: {co.most_common(8)}")

    print("== 第6.7步：词对同现评论数 ==")
    for a, b in [("screen", "battery"), ("phone", "battery"), ("good", "bad")]:
        n = sum(1 for s in sentences if a in s and b in s)
        print(f"同时包含 {a} 和 {b} 的评论：{n} 条")

    print("== 第7步：导出三件套 ==")
    export_artifacts(model2, OUT_DIR, window=2)   # window=2 只留作对照
    export_artifacts(model5, OUT_DIR, window=5)   # ← 交给实验2 的用 window=5（老师要求统一）
    print("全部完成，去 outputs/my_exp01 里看产出文件")