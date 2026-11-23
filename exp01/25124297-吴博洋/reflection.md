# 实验1：词向量表示与相似度分析

学号：25124297　姓名：吴博洋

## 个人感想

我通过这个实验了解了如何为一个文件创造一个虚拟python环境，在这里是.venv312。我也通过Amazon上的评论运行并训练了window=2和window=5两个词向量模型，整理了两组窗口的相似度结果。在异常分析中，我重点看了phone和grill、sound和calender的词频、评论覆盖情况和上下文，这让我理解了词向量接近只能反映上下文分布相似，不能等同于两个词意思相近。这个main.py可以帮助识别低频或语境单一的词,并且量化了2，5两个窗口变化带来的差异，可以用于查询、比较和分析。

## 核心代码与运行方法

[main.py](./main.py) 在课程参考实现基础上，读取window=2和window=5的已有词向量，比较同一查询词的Top-5近邻，统计查询词及近邻的出现次数、覆盖评论条数，并计算两组近邻的Jaccard重合度。词表外的词明确输出OOV。分词、CSV读取和文件摘要复用课程的 [lab_common.py](../lab_common.py)，训练使用课程的 [run_lab.py](../run_lab.py)。

使用Python 3.12及课程的 [requirements.txt](../requirements.txt)。在上一级实验目录打开PowerShell，使用已配置的环境运行：

```powershell
& "./.venv312/Scripts/python.exe" "./25124297-吴博洋/main.py" phone sound service calender zzunknownword
```

程序默认读取上一级实验目录的训练CSV和两个窗口模型，输出本文件夹中的“观察结果.md”。数据与模型准备方法见 [课程实验说明](../README.md)。使用其他数据或模型时，通过命令行指定路径：

```powershell
& "./.venv312/Scripts/python.exe" "./25124297-吴博洋/main.py" phone sound --model-dir "./outputs/formal" --train-csv "./data/train_sample.csv"
```

程序会核对训练CSV摘要与模型训练记录，保证词频统计和近邻查询使用同一份语料。运行时不传查询词即可交互输入，直接回车默认查询phone、sound和service。

## 已有结果中的观察

在固定的5,000条训练评论上，phone在两个窗口的Top-5中只共享grill，近邻集合重合度为11.11%，说明窗口变化会改变近邻名单；sound在window=2的第一近邻是calender，但该词只出现3次，说明低频词的向量可能缺少足够的语境支持。
