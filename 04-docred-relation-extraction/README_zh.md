# 实验4：DocRED 文档级关系抽取

## 概述

本实验将预训练语言模型适配到 DocRED 这一大规模文档级关系抽取任务上。
DocRED 中大部分关系需要从多个句子中联合抽取，因此模型需要具备获取和
综合整篇文档信息的能力，尤其是跨句关系抽取。模型基于 BERT，并扩展到
RoBERTa，在不同训练数据规模下进行评测。

## 我做了哪些工作

- 将课程提供的 BERT 框架适配到 DocRED 任务，完整跑通训练和评测流程。
- 分别用 10%、50%、100% 训练数据训练模型（各 100 epoch），研究训练
  数据规模对性能的影响。
- 在相同超参下训练 RoBERTa 进行对比。
- 将 `batch_size` 调整为 4，配合 `gradient_accumulation_steps = 8`，
  以适配 GPU 显存。
- 对 `config/Config.py` 做了小幅修改：增加 checkpoint 保存和一些
  实现细节调整。
- 绘制训练损失曲线，对比不同模型和不同数据量下的收敛情况。
- 在测试集上为 BERT（100%）和 RoBERTa（100%）分别推理，生成
  `bert_model_100p.pt_test_index.json` 和
  `roberta_model_100p.pt_test_index.json`。

## 关键结果

开发集结果（Threshold / Best F1 / AUC / Ign_F1 / Ign_AUC）：

| 模型 | 训练数据 | Theta | Best F1 | AUC | Ign_F1 | Ign_AUC |
|------|----------|-------|---------|-----|--------|---------|
| BERT | 10% | 0.4890 | 0.4249 | 0.3665 | 0.4004 | 0.3391 |
| BERT | 50% | 0.7367 | 0.5332 | 0.5090 | 0.5101 | 0.4763 |
| BERT | 100% | 0.8666 | 0.5662 | 0.5534 | 0.5439 | 0.5209 |
| **RoBERTa** | **100%** | **0.8949** | **0.5731** | **0.5611** | **0.5510** | **0.5278** |

最佳结果：**RoBERTa + 100% 训练集，Best F1 = 0.5731**。

## 结果分析

- 训练数据越多，性能越好；50% 已接近 100%，计算资源有限时 50% 是
  合理选择。
- 相同超参下，RoBERTa 优于 BERT。
- RoBERTa 的训练 loss 高于 BERT，但曲线更稳定，说明增加 epoch
  可能进一步提升 RoBERTa。

## 文件清单

- `README.md` — 英文版说明
- `README_zh.md` — 本文件（中文版）
- `report.pdf` — 完整实验报告（PDF）

## 数据集（不包含）

- **DocRED** — 大规模文档级关系抽取数据集。每篇文档包含多个句子，
  标注了实体对及其关系。训练集中只出现部分关系；Ign-F1 / Ign-AUC
  用于评估模型对未见过关系的泛化能力。

> 完整数据集不包含在本仓库中。

## 技术栈

Python、PyTorch、HuggingFace Transformers（BERT、RoBERTa）、NumPy、
scikit-learn（PR-AUC）、Matplotlib