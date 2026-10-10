# NLP Course Projects

## Overview

This repository is a collection of six NLP course projects completed during 
the Tsinghua NLP Training Camp. It is a **portfolio for demonstration**, 
not a runnable package: the goal is to show how I framed each task, what 
methods I used, what results I obtained, and how I analyzed them.

**What is included**: per-experiment README (English + Chinese), the 
experiment report, and (for some experiments) the notebooks or scripts 
I wrote.

**What is not included**: course-provided code, full datasets, and 
trained model weights. The experiments cannot be reproduced end-to-end 
from this repository alone.

Each sub-experiment README explains the task, my contributions, key 
results, datasets, and tech stack.

**Language note**: All experiment reports (`report.pdf`) and code comments 
in this repository are written in **Chinese**, as the course was delivered 
in Chinese. The READMEs are provided in both English and Chinese.

For detailed content of each experiment, please refer to its own README 
(e.g. `01-word2vec-transe/README.md`).

## Repository Structure

```
nlp-course-projects/
├── README.md # English main README
├── README_zh.md # Chinese main README
├── .gitignore
├── 01-word2vec-transe/
├── 02-seq2seq-translation/
├── 03-sentiment-analysis/
├── 04-docred-relation-extraction/
├── 05-judicial-reading-comprehension/
└── 06-covid-rumor-detection/
```

## Experiments at a Glance

| # | Experiment | Task | Key Result | Tech Stack |
|---|------------|------|-----------|-----------|
| 1 | Word2Vec & TransE | Word embeddings + knowledge graph embeddings | Word2Vec Spearman 0.69 on WordSim353; TransE trained to loss 0.46 | Gensim, PyTorch, NumPy |
| 2 | seq2seq Translation | Chinese–English NMT | Corpus BLEU 0.15 → 0.77 | PyTorch, NLTK |
| 3 | Sentiment Analysis | Binary classification on Rotten Tomato | BERT + SVM Accuracy 79.32% | scikit-learn, gensim, Transformers |
| 4 | DocRED | Document-level relation extraction | RoBERTa Best F1 0.5731 | PyTorch, Transformers |
| 5 | Judicial Reading Comprehension | Multi-hop QA on Chinese judicial documents | RoBERTa Answer F1 74.20%, Joint F1 57.36% | PyTorch, Transformers |
| 6 | COVID-19 Rumor Detection | Binary classification of rumors vs. facts | BERT Accuracy 95.2% | scikit-learn, SnowNLP, Transformers |

## Skills Demonstrated

- **Model evaluation**: Spearman correlation, BLEU, EM / F1, Ign-F1 / AUC, 
  joint EM / F1, Accuracy / Precision / Recall / F1 / AUC.
- **Multi-model comparison**: comparing BERT vs. RoBERTa, LR vs. RF, 
  feature-based vs. neural methods under identical settings.
- **Problem diagnosis**: identifying overfitting, class imbalance, 
  mismatch between pre-trained models and target domains, and 
  embedding norm issues; proposing and testing concrete fixes.
- **Data processing**: building balanced and imbalanced datasets from 
  raw sources, deduplication, filtering, feature engineering.
- **Transfer learning**: adapting BERT / RoBERTa to relation extraction, 
  judicial QA, and rumor detection.
- **Hyperparameter analysis**: systematic sweeps and their effect on 
  performance (Word2Vec, seq2seq).

## Notes

This repository is a demonstration of my work, not a reproducible codebase. 
Course-provided code and datasets are not redistributed. Experiment reports 
and code comments are in Chinese; each sub-experiment README is available 
in both English and Chinese.