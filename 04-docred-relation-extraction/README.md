# Experiment 4: DocRED Document-Level Relation Extraction

## Overview

This experiment adapts pre-trained language models to DocRED, a 
large-scale document-level relation extraction task. Most relations in 
DocRED require joint extraction from multiple sentences, so the model 
must aggregate information across the whole document, especially 
cross-sentence relations. The model is built on BERT and extended to 
RoBERTa, and evaluated under different training-data sizes.

## What I Did

- Adapted the provided BERT-based framework to the DocRED task and ran 
  training and evaluation end to end.
- Trained the model with 10%, 50%, and 100% of the training data 
  (100 epochs each) to study the effect of training-data size.
- Trained RoBERTa under the same hyperparameters for comparison.
- Adjusted `batch_size` to 4 with `gradient_accumulation_steps = 8` to 
  fit GPU memory.
- Made small modifications to `config/Config.py` to add checkpoint saving 
  and minor implementation details.
- Plotted training loss curves to compare convergence across models and 
  data sizes.
- Ran inference on the test set for both BERT (100%) and RoBERTa (100%), 
  producing `bert_model_100p.pt_test_index.json` and 
  `roberta_model_100p.pt_test_index.json`.

## Key Results

Development set results (Threshold / Best F1 / AUC / Ign_F1 / Ign_AUC):

| Model | Train % | Theta | Best F1 | AUC | Ign_F1 | Ign_AUC |
|-------|---------|-------|---------|-----|--------|---------|
| BERT | 10% | 0.4890 | 0.4249 | 0.3665 | 0.4004 | 0.3391 |
| BERT | 50% | 0.7367 | 0.5332 | 0.5090 | 0.5101 | 0.4763 |
| BERT | 100% | 0.8666 | 0.5662 | 0.5534 | 0.5439 | 0.5209 |
| **RoBERTa** | **100%** | **0.8949** | **0.5731** | **0.5611** | **0.5510** | **0.5278** |

Best result: **RoBERTa + 100% training data, Best F1 = 0.5731**.

## Analysis

- More training data consistently improves performance; 50% is already 
  close to 100%, so 50% is a reasonable choice under limited compute.
- Under the same hyperparameters, RoBERTa outperforms BERT.
- RoBERTa has a higher training loss than BERT but more stable curves, 
  suggesting that more epochs could further improve RoBERTa.

## Files

- `README.md` — this file (English)
- `README_zh.md` — Chinese version
- `report.pdf` — full experiment report (PDF)

## Datasets (not included)

- **DocRED** — a large-scale document-level relation extraction dataset. 
  Each document contains multiple sentences and annotated entity pairs 
  with relations. Only a subset of relations appears in the training set; 
  the Ign-F1 / Ign-AUC metrics evaluate generalization to unseen relations.

> Full dataset is not included in this repository.

## Tech Stack

Python, PyTorch, HuggingFace Transformers (BERT, RoBERTa), NumPy, 
scikit-learn (PR-AUC), Matplotlib