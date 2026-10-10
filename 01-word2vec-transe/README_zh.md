# 实验1：Word2Vec 与 TransE

## 概述

本实验涵盖两种经典的表示学习模型：用于词向量的 Word2Vec，以及用于
知识图谱嵌入的 TransE。我在完成基本要求之外，重点关注两件事：
**超参数如何影响词向量质量**，以及**训练完成的模型为何在简单推理任务上
仍然失败**。

对于 Word2Vec，我把课程提供的训练脚本和评估脚本合并成一条流水线，
并做了三组参数扫描，观察词向量维度、窗口大小、最小出现次数对 
WordSim353 上 Spearman 相关系数的影响。

对于 TransE，在补全课程骨架中缺失的两个函数之后，我额外实现了一套
推理与诊断工具：任务1（头实体 + 关系 → 尾实体）和任务2（头实体 + 
尾实体 → 关系），分析了实体向量的范数分布，发现训练得到的向量范数
异常偏小。随后我在推理阶段手动对向量做 L2 归一化，并证明这显著
提升了正确答案的排名。

## 我做了哪些工作

### Word2Vec
- 运行课程提供的 `word2vec.py` 和 `evaluate.py`，得到 WordSim353 上的
  基线 Spearman 相关系数（≈0.69）。
- 在 notebook 中实现 `main_function(vector_size, window, min_count)` 和 
  `plot_spcors(x, y, xlabel)`，把训练和评估合并为一条流水线。
- 完成三组参数扫描：
  - `vector_size`：50 到 500（步长50）
  - `window`：2 到 20（步长2）
  - `min_count`：2 到 20（步长2）以及 20 到 100（步长10）

### TransE
- 补全课程提供的 `TransE.py` 骨架中的 `_calc()` 和 `loss()`。
- 训练500轮。
- 实现 `load_id_mapping()` 和 `load_vectors()`，用于加载 ID 映射和向量。
- 实现 `task1_find_closest_tails()` 和 `task2_find_closest_relations()` 
  用于推理和验证。
- 实现 `analyze_embeddings_quality()`，检查向量范数和数值分布。
- 实现 `task1_with_normalization()`，在推理时手动 L2 归一化，并重新
  执行任务1。

## 关键结果

### Word2Vec（WordSim353 上的 Spearman 相关系数）

| 设置 | Spearman |
|------|----------|
| 基线（vector_size=200, window=10, min_count=10） | ≈0.69 |
| 最佳（vector_size=400） | 0.699 |

- `vector_size` 在 100–200 之间较为合理，继续增大没有明显提升。
- `window` 在 10–15 之间较为合理；小于 10 时相关系数急剧下降。
- `min_count` 在 20–100 范围内与效果无明显相关性。

### TransE

- 训练500轮，损失从 91 降至 0.46。
- 任务1（Q30 + P36 → 尾实体）：期望的 Q61（Washington, D.C.）未进入
  Top10，Top1 甚至是头实体 Q30 本身。
- 任务2（Q30 + Q49 → 关系）：期望的 P30（continent）未进入 Top5。
- 在推理时手动做 L2 归一化后：**Q61 的排名提升至 85**。

## 分析

- **Word2Vec**：Spearman 相关系数很快饱和。词向量维度超过 100–200 
  之后不再有明显收益，说明在 Text8 语料和 WordSim353 评估集上，
  该模型家族已经接近上限。窗口大小影响更大：窗口太小会漏掉长距离
  语义关系，10–15 是一个较好的折中。
- **TransE**：损失确实下降了，但推理结果明显错误。我的诊断函数
  （`analyze_embeddings_quality`）显示，实体向量的平均 L2 范数只有
  ≈0.046，异常偏小。这说明模型没有执行 TransE 所假设的归一化约束
  （`||h|| = ||r|| = ||t|| = 1`）。我没有重新训练，而是在推理阶段
  手动归一化向量并重跑任务1。正确答案 Q61 从榜上无名提升到第85名，
  验证了问题出在归一化，而不是模型学到的结构。

## 文件清单

- `README.md` — 英文版说明
- `README_zh.md` — 本文件（中文版）
- `report.ipynb` — 完整实验报告（Word2Vec + TransE）
- `report.pdf` — 报告的 PDF 版本

> 所有代码（Word2Vec 参数分析、TransE `_calc()`/`loss()` 补全、
> 任务1/2推理、向量质量分析、归一化实验）均在 `report.ipynb` 中实现，
> 不包含单独的 `.py` 文件。

## 数据集（不包含）

- **Text8** — 用于 Word2Vec 训练，约100 MB 的清洗后 Wikipedia 文本。
- **WordSim353** — 词语相似度基准数据集，353 个词对及人工相似度打分，
  用于通过 Spearman 相关系数评估 Word2Vec。
- **Wikidata** — 用于 TransE 训练，本实验使用 50,000 个实体和 378 个关系，
  每个三元组为 (头实体, 关系, 尾实体)。

> 完整数据集和训练得到的向量文件不包含在本仓库中。

## 技术栈

Python、Gensim（Word2Vec）、NumPy、SciPy（Spearman）、Matplotlib、
PyTorch（TransE）