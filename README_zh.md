# NLP 课程项目

## 概述

本仓库是清华 NLP 训练营六个实验的集合。它的定位是**成果展示**，而不是
一个可运行的完整项目：重点在于展示我如何理解每个任务、采用了什么方法、
得到了什么结果，以及如何分析这些结果。

**包含的内容**：每个实验的 README（中英双语）、实验报告，以及部分实验
中我编写的 notebook 或脚本。

**不包含的内容**：课程提供的代码、完整数据集、训练好的模型权重。
本仓库无法单独端到端复现所有实验。

每个子实验的 README 分别说明了任务、我的贡献、关键结果、数据集和技术栈。

**语言说明**：本仓库中的实验报告（`report.pdf`）和代码注释均为**中文**，
因为课程本身以中文授课。各 README 提供中英双语版本。

每个实验的详细内容，请参考对应子实验的 README（如 
`01-word2vec-transe/README_zh.md`）。

## 仓库结构
nlp-course-projects/
├── README.md # 英文主 README
├── README_zh.md # 中文主 README
├── .gitignore
├── 01-word2vec-transe/
├── 02-seq2seq-translation/
├── 03-sentiment-analysis/
├── 04-docred-relation-extraction/
├── 05-judicial-reading-comprehension/
└── 06-covid-rumor-detection/


## 六个实验总览

| # | 实验 | 任务 | 关键结果 | 技术栈 |
|---|------|------|----------|--------|
| 1 | Word2Vec & TransE | 词向量 + 知识图谱向量 | Word2Vec 在 WordSim353 上 Spearman 0.69；TransE 训练后损失 0.46 | Gensim、PyTorch、NumPy |
| 2 | seq2seq 翻译 | 中英神经机器翻译 | Corpus BLEU 0.15 → 0.77 | PyTorch、NLTK |
| 3 | 情感分析 | Rotten Tomato 二分类 | BERT + SVM 准确率 79.32% | scikit-learn、gensim、Transformers |
| 4 | DocRED | 文档级关系抽取 | RoBERTa Best F1 0.5731 | PyTorch、Transformers |
| 5 | 司法阅读理解 | 中文司法文书多步推理问答 | RoBERTa Answer F1 74.20%，Joint F1 57.36% | PyTorch、Transformers |
| 6 | 疫情谣言检测 | 谣言 vs 事实二分类 | BERT 准确率 95.2% | scikit-learn、SnowNLP、Transformers |

## 技能展示

- **模型评估**：Spearman 相关系数、BLEU、EM / F1、Ign-F1 / AUC、
  Joint EM / F1、Accuracy / Precision / Recall / F1 / AUC。
- **多模型对比**：在相同设置下对比 BERT vs. RoBERTa、逻辑回归 vs.
  随机森林、特征方法 vs. 神经网络方法。
- **问题诊断**：识别过拟合、类别不平衡、预训练模型与目标领域不匹配、
  向量范数异常等问题，并提出和验证改进方案。
- **数据处理**：从原始数据构建平衡与不平衡数据集、去重、过滤、特征工程。
- **迁移学习**：将 BERT / RoBERTa 适配到关系抽取、司法问答、谣言检测任务。
- **超参分析**：系统性的参数扫描及其对性能的影响（Word2Vec、seq2seq）。

## 说明

本仓库为个人成果展示，不是可复现的完整代码库。课程提供的代码和数据集
不再分发。实验报告和代码注释为中文；各子实验 README 提供中英双语。