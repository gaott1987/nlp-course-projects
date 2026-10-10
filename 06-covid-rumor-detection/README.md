# Experiment 6: COVID-19 Rumor Detection

## Overview

This was an open-ended social computing assignment: given several types of 
COVID-19 related social data (rumors, Chinese news, and legal documents), 
I had to define a concrete task myself, pick a dataset, and choose a 
method. Among the three data sources, I chose the **rumor dataset 
(CSDC-Rumor)** for two reasons:

1. Rumor detection is a well-defined binary classification problem with 
   clear ground-truth labels, which makes objective evaluation possible.
2. I had already worked on sentiment analysis (news) and judicial reading 
   comprehension (legal documents) in previous experiments, so rumor 
   detection was the only direction that would let me explore a **new 
   task** rather than repeat an existing pipeline.

For the method, I surveyed the three common rumor detection paradigms 
(feature-based, neural, and propagation-based) and rejected the 
propagation-based approach because the dataset does not contain complete 
retweet / reply chains. I then built two lines of attack and compared 
them directly:

- **Feature-based**: 12 handcrafted features + Logistic Regression / 
  Random Forest.
- **Neural**: BERT fine-tuning.

A key focus of this experiment was studying **how class imbalance affects 
the two method families**, since the full dataset is heavily skewed 
towards rumors (331 vs. 80 facts), while the balanced subset is not.

## What I Did

- Analyzed the CSDC-Rumor dataset and reconstructed two datasets from 
  it:
  - **Balanced**: 62 rumors vs. 61 facts.
  - **Full**: 331 rumors vs. 80 facts.
- Designed a 12-dimensional feature set covering vocabulary, style, and 
  surface statistics:
  - Vocabulary: uncertainty words, extreme-emotion words, fact words.
  - Style: length, exclamation count, question count, digit density, 
    first-person count, average sentence length, sentiment, capital 
    ratio, ellipsis count.
- Built the feature-based pipeline with Z-score normalization and 
  compared Logistic Regression and Random Forest on both datasets.
- Analyzed feature importance and how it shifts between the balanced and 
  full datasets.
- Fine-tuned BERT on the full dataset (lr=2e-5, 3 epochs) and compared 
  it against the feature-based models.

## Key Results

### Balanced dataset (62 rumors vs. 61 facts, 25 test samples)

| Model | Accuracy | AUC |
|-------|----------|-----|
| Logistic Regression | 0.92 | 0.929 |
| Random Forest | 0.88 | 0.936 |

### Full dataset (331 rumors vs. 80 facts, 83 test samples)

| Model | Accuracy | AUC |
|-------|----------|-----|
| Logistic Regression | 0.80 | — |
| Random Forest | 0.93 | — |
| **BERT** | **0.952** | — |

Best result: **BERT, Accuracy 95.2%**.

## Analysis

- On the balanced dataset, Logistic Regression outperforms Random Forest 
  (0.92 vs. 0.88), suggesting that the 12 features are roughly linearly 
  separable and LR is more robust to feature noise.
- On the full dataset, the ranking reverses: Random Forest reaches 0.93 
  while LR drops to 0.80. LR is more sensitive to class imbalance and 
  sacrifices minority-class precision for recall; RF handles imbalance 
  better through ensembling and class weights.
- Feature importance:
  - Balanced: `length`, `avg_sentence_length`, `sentiment` rank top-3.
  - Full: `length`, `avg_sentence_length`, `capital_ratio` rank top-3. 
    `capital_ratio` becomes important because Weibo rumors rarely contain 
    English terms, while factual text often includes acronyms 
    (e.g. COVID-19, FDA). `exclamation_count` also rises sharply, 
    matching the intuition that Weibo rumors use exclamation marks to 
    incite emotion.
  - `uncertainty_count` and `extreme_count` are consistently unimportant, 
    likely because factual reports also contain these words 
    (e.g. "deaths" in official reports).
- BERT on the full dataset reaches 95.2% accuracy, 2.4 points above the 
  best feature-based model. It notably improves the minority class 
  (facts): recall rises from 0.75 to 0.94, and F1 from 0.80 to 0.88, 
  reducing the risk of misclassifying real information as rumors.

## Files

- `README.md` — this file (English)
- `README_zh.md` — Chinese version
- `report.pdf` — full experiment report (PDF)
- `rumor_features_balance.ipynb` — feature-based method on the balanced 
  dataset (LR + RF)
- `rumor_features_unbalance.ipynb` — feature-based method on the full 
  dataset (LR + RF)
- `rumor_bert.ipynb` — BERT fine-tuning on the full dataset

## Datasets (not included)

- **CSDC-Rumor** (THUNLP, https://covid19.thunlp.org/):
  - **Weibo rumor data** — 324 entries, 273 with non-empty rumor text.
  - **Tencent / DXY rumor–fact pairs** — 124 entries with both rumor 
    text and factual abstract, labeled by `explain` type 
    (pseudo-science, undetermined, true, pseudo-common-sense).

> The full dataset is not included in this repository.

## Tech Stack

Python, scikit-learn (LR, RF, Z-score, metrics), SnowNLP (sentiment), 
HuggingFace Transformers (BERT), NumPy, pandas, Matplotlib, seaborn