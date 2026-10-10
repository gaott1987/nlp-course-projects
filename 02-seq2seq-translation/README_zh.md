# 实验2：seq2seq 神经机器翻译

## 概述

我基于 seq2seq 框架构建了一个中英神经机器翻译系统，从基础的 RNN 编码器-
解码器出发，逐步改进。我的工作包括：

- 修改课程提供的骨架（特殊字符索引、低频词过滤、划分验证集）。
- 推理调优：带重复惩罚和长度惩罚的 BeamSearch。
- 训练调优：超参分析与调整。
- 结构改进：GRU 和 Attention（双向 GRU 编码器 + 解码器 attention）。
- 批处理训练，提高 GPU 利用率。
- 用最佳配置完成最终测试集推理。

## 我做了什么

### 基础模型
- 在课程提供骨架的基础上修改：
  - 调整特殊字符索引：`<SOS>=1`、`<EOS>=2`，`0` 保留给 padding，
    新增 `<UNK>=3`。
  - 增加低频词过滤（`min_count=2`）。
  - 从训练数据中划分 10% 作为验证集，用于计算 BLEU。

### 推理调优
- 实现 BeamSearch，并加入两个约束：
  - 重复惩罚（防止重复输出）。
  - 长度惩罚（鼓励生成长度适中的句子）。

### 训练调优
- 分析原始超参（`hidden_size=500`、`n_layers=2`、`dropout_p=0.05`、
  `learning_rate=0.0001`）。
- 尝试将 `dropout_p` 提高到 0.2、`learning_rate` 提高到 0.0005，
  并将 `teacher_forcing_ratio` 降到 0.3。

### 结构改进
- **GRU**：将编码器和解码器中的 RNN 替换为 GRU。
- **Attention**：在 GRU 模型基础上，编码器采用双向 GRU，
  解码器加入 attention 机制。为加快训练，`hidden_size` 降为 256。

### 批处理训练
- 重写训练循环，支持批处理，提高 GPU 利用率。
- 尝试两套超参：
  - (A) `hidden_size=256`、`n_layers=2`、`dropout=0.1`、`lr=0.001`、
    `batch_size=32`、`n_epochs=20`。
  - (B) `hidden_size=512`、`n_layers=2`、`dropout=0.2`、`lr=0.001`、
    `batch_size=32`、`n_epochs=10`。
- 选择 (A) 进行最终测试集推理，并在其基础上应用 BeamSearch。

## 关键结果

| 模型 | Corpus BLEU | 平均句子 BLEU |
|------|-------------|----------------|
| 基础 RNN | 0.1519 | 0.1411 |
| 基础 + BeamSearch | 0.1263 | 0.1188 |
| 基础 + 超参调优 | 0.1174 | 0.1058 |
| GRU | 0.3231 | 0.3140 |
| Attention（双向GRU + attention） | 0.4123 | 0.3983 |
| 批处理 Attention (A) | 0.7496 | 0.7137 |
| 批处理 Attention (B) | 0.6840 | 0.6450 |
| 批处理 Attention (A) + BeamSearch | **0.7684** | **0.7362** |

最终最佳：**Corpus BLEU 0.7684**（批处理 Attention + BeamSearch）。

## 分析

- **基础 RNN**：只能翻译出少数高频词，并在句末生成大量句号。
  Corpus BLEU 0.1519。
- **BeamSearch**：重复和过短输出减少了，但 BLEU 降到 0.1263 —— 瓶颈在
  解码器本身的能力，而不是搜索策略。
- **超参调优**：提高 `learning_rate` 后损失过早收敛，BLEU 降到 0.1174。
- **诊断**：由于推理调优和训练调优都未奏效，我判断瓶颈在 RNN 结构本身，
  转向结构改进。
- **GRU**：BLEU 提升到 0.3231 —— 用门控单元替换 RNN 效果明显。
- **Attention**：BLEU 达到 0.4123（比 GRU 提高约27%）。损失曲线平滑度
  排序为 Attention > GRU > RNN。
- **批处理训练**：训练效率大幅提高，模型得以充分收敛。批处理 Attention (A)
  达到 BLEU 0.7496 —— 相比未批处理的 Attention 模型（0.4123）提升很大。
- **在 (A) 上应用 BeamSearch**：BLEU 0.7684，略优于贪心。

## 文件清单

- `README.md` — 英文版说明
- `README_zh.md` — 本文件（中文版）
- `report.pdf` — 完整实验报告（PDF）
- `seq2seq-translation_base.ipynb` — 基础 RNN 模型
  （基于课程骨架修改）
- `seq2seq-translation_gru.ipynb` — GRU 模型
- `seq2seq-translation_attention.ipynb` — Attention + 批处理
- `test_translations_batch.txt` — 测试集原文和模型推理译文
  （批处理 Attention 模型）

## 数据集（不包含）

- **NIST 中英平行语料** — 90,000 句对，裁剪到 68,898 句对。
  - 训练集：62,009（90%）
  - 验证集：6,889（10%）
  - 输入词汇表：3,883
  - 输出词汇表：8,855

> 完整数据集不包含在本仓库中，仅提供报告、notebook 和最终测试集译文
> 用于展示目的。

## 技术栈

Python、PyTorch、NLTK（BLEU）、Matplotlib