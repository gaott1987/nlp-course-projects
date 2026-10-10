# 实验5：司法阅读理解

## 概述

本实验在中文司法文书上构建了一个基于 BERT 的端到端问答系统。数据集覆盖
民事、刑事、行政三类案件，共约 5,100 个问答对（每类约 1,700 个）。每个
问题都需要跨多个句子进行推理，而不是单句就能回答。模型遵循 HotpotQA 风格
的设置：给定案件描述和问题，模型需要同时给出答案和答案依据（支持事实）。

## 我做了哪些工作

- 阅读并注释了课程提供的 baseline 代码（`data_process.py`、
  `run_cail.py`、`modeling.py`）以及自行下载的 `evaluate.py`，为数据结构、
  特征转换、多任务模型和评估逻辑添加了理解注释。
- 编写 `plot_curves.py`，对每个 epoch 的中间预测文件进行评估，并绘制
  EM / F1 / Joint 指标随 epoch 变化的曲线，用于挑选最佳 epoch。
- 训练基准模型 `chinese-bert-wwm` 10 个 epoch，并分析每个 epoch 的评估结果。
- 在相同设置下训练 `chinese-roberta-wwm-ext`，与基准模型对比。
- 另外尝试了 THUNLP 的司法 BERT（刑事、百度百科两个版本）并对比结果。
- 对比不同模型的收敛速度，并分析答案预测与支持事实预测之间的差距。

## 关键结果

| 模型 | 最佳 Answer F1 | 最佳 Answer EM | 最佳 SP F1 | 最佳 Joint F1 |
|------|---------------|----------------|-----------|---------------|
| chinese-bert-wwm（基准） | 73.60% (E7) | 65.08% (E10) | 70.04% (E10) | 55.58% (E10) |
| **chinese-roberta-wwm-ext** | **74.20% (E7)** | **66.87% (E7)** | **72.80% (E10)** | **57.36% (E10)** |
| 司法BERT（刑事） | 67.70% (E10) | 59.33% (E10) | 63.41% (E10) | 46.25% (E10) |
| 司法BERT（百度百科） | 54.44% (E8) | 45.63% (E7) | 57.15% (E9) | 36.69% (E9) |

最佳结果：**chinese-roberta-wwm-ext，Answer F1 74.20%，Joint F1 57.36%**。

## 结果分析

- 基准模型 `chinese-bert-wwm` 表现已经很好（Answer F1 > 73%，
  EM > 65%），且第 7 到第 10 个 epoch 的 Answer F1 基本稳定。
- `chinese-roberta-wwm-ext` 优于基准模型，收敛更快，体现了 RoBERTa
  的架构优势。
- 司法 BERT（刑事）效果不如基准，因为实验数据是民事、刑事、行政的混合，
  其中三分之二不是刑事文本，主要在刑事文本上预训练的模型迁移效果不好。
- 司法 BERT（百度百科）表现最差，其预训练引入了更多通用、甚至不专业的
  知识。
- 所有模型中，支持事实（SP）的收敛速度都慢于答案预测，说明找到正确的
  依据比给出正确答案更难。

## 文件清单

- `README.md` — 英文版说明
- `README_zh.md` — 本文件（中文版）
- `report.pdf` — 完整实验报告（PDF）
- `plot_curves.py` — 评估中间 checkpoint 并绘制 EM / F1 / Joint 随
  epoch 变化曲线的脚本

> baseline 代码（`data_process.py`、`run_cail.py`、`modeling.py`）和
> `evaluate.py` 由课程提供（或从官方来源下载），不包含在本仓库中；
> 仅包含我编写的 `plot_curves.py`。

## 数据集（不包含）

- **司法阅读理解数据集** — 约 5,100 个多步推理问答对，覆盖民事、刑事、
  行政三类（每类约 1,700 个），按 9:1 划分为训练集和开发集。每条样本包含
  案件描述、问题、答案和支持事实。

> 完整数据集不包含在本仓库中。课程明确要求该数据不得传播。

## 技术栈

Python、PyTorch、HuggingFace Transformers（BERT、RoBERTa）、NumPy、
Matplotlib