在做实验一的时候我遇到了两个问题。
一个是一开始我以为 `read_reviews` 返回的是一堆词。读了才发现它返回的是 `Review` 对象，要再调 `.tokens()` 才拿到词列表。中间那个 `rows` 变量我用了一次就没再用过.所以读代码的时候得在几个文件来回跳,才能确认一条评论最后变成了什么。这个流程读起来真的很绕。
第二个问题是重跑时报错：
Output directory is not empty; choose a new run directory
我后来查了源码才知道是它故意的
`if a.output_dir.exists() and any(a.output_dir.iterdir()): raise ValueError(...)`，防止一次重跑把上一次的结果覆盖掉。
我还做了点代码改动，把实验结果直接打印到终端。
把
```python
write_table(a.output_dir / "word_pairs.csv", list(pairs[0]), pairs)
```
改成了
```python
print(f"{'词对':<16}{'w=2':>8}{'w=5':>8}{'变化':>9}")
for left, right in PAIRS:
    s2, s5 = models[2].wv.similarity(left, right), models[5].wv.similarity(left, right)
    print(f"{left + '-' + right:<18}{float(s2):>8.4f}{float(s5):>8.4f}{float(s5 - s2):>+9.4f}")
```
结果只有 5 行，落盘再打开看，比直接看屏幕多两步。而且不写文件就不用建输出目录，上面那个「目录非空」的报错也就不会再出现。 
把结果存下来和看清结果是两件事。原代码必须落盘，因为实验二要用 `embedding.npy` 和 `vocab.json`。

在这次实验中我学到最多的是如何装载环境和实验的基本流程，以及阅读大段代码的能力。
