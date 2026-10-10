# Experiment 5: Judicial Reading Comprehension

## Overview

This experiment builds a BERT-based end-to-end question answering system 
on Chinese judicial documents. The dataset covers civil, criminal, and 
administrative cases, with about 5,100 multi-hop QA pairs in total 
(~1,700 each). Each question requires reasoning across multiple 
sentences rather than a single sentence. The model follows the 
HotpotQA-style setup: given a case description and a question, the model 
must produce both an answer and the supporting facts.

## What I Did

- Read and annotated the provided baseline code 
  (`data_process.py`, `run_cail.py`, `modeling.py`) and the 
  `evaluate.py` script, adding comments to explain data structures, 
  feature conversion, the multi-task model, and the evaluation logic.
- Wrote `plot_curves.py` to evaluate intermediate checkpoints 
  (one prediction file per epoch) and plot the EM / F1 / Joint metrics 
  as curves over epochs, in order to pick the best epoch.
- Trained the baseline model `chinese-bert-wwm` for 10 epochs and 
  analyzed the per-epoch evaluation results.
- Trained `chinese-roberta-wwm-ext` under the same settings and 
  compared it against the baseline.
- Also tried the THUNLP judicial BERT (criminal and Baidu-Baike 
  variants) and compared the results.
- Compared convergence speed across models and analyzed the gap between 
  answer prediction and supporting-fact prediction.

## Key Results

| Model | Best Answer F1 | Best Answer EM | Best SP F1 | Best Joint F1 |
|-------|---------------|----------------|-----------|---------------|
| chinese-bert-wwm (baseline) | 73.60% (E7) | 65.08% (E10) | 70.04% (E10) | 55.58% (E10) |
| **chinese-roberta-wwm-ext** | **74.20% (E7)** | **66.87% (E7)** | **72.80% (E10)** | **57.36% (E10)** |
| Judicial BERT (criminal) | 67.70% (E10) | 59.33% (E10) | 63.41% (E10) | 46.25% (E10) |
| Judicial BERT (Baidu Baike) | 54.44% (E8) | 45.63% (E7) | 57.15% (E9) | 36.69% (E9) |

Best result: **chinese-roberta-wwm-ext, Answer F1 74.20%, Joint F1 57.36%**.

## Analysis

- The baseline `chinese-bert-wwm` already performs well 
  (Answer F1 > 73%, EM > 65%), and Answer F1 is basically stable 
  from epoch 7 to 10.
- `chinese-roberta-wwm-ext` improves over the baseline and converges 
  faster, showing the architectural advantage of RoBERTa.
- The judicial BERT (criminal) performs worse than the baseline, 
  because the training data is a mix of civil, criminal, and 
  administrative cases, and two-thirds of it is not criminal. 
  A domain-specific model trained mainly on criminal text does not 
  transfer well.
- The Baidu-Baike judicial BERT performs the worst; its pre-training 
  appears to have introduced more general, less professional knowledge.
- Across all models, supporting-fact prediction converges more slowly 
  than answer prediction, suggesting that finding the correct evidence 
  is harder than producing the correct answer.

## Files

- `README.md` — this file (English)
- `README_zh.md` — Chinese version
- `report.pdf` — full experiment report (PDF)
- `plot_curves.py` — script for evaluating intermediate checkpoints and 
  plotting EM / F1 / Joint curves over epochs

> The baseline code (`data_process.py`, `run_cail.py`, `modeling.py`) 
> and `evaluate.py` are provided by the course (or downloaded from the 
> official source). They are not included in this repository; only the 
> `plot_curves.py` script I wrote is included.

## Datasets (not included)

- **Judicial reading comprehension dataset** — about 5,100 multi-hop QA 
  pairs across civil, criminal, and administrative cases (~1,700 each), 
  split 9:1 into train and dev. Each sample contains a case description, 
  a question, an answer, and supporting facts.

> Full dataset is not included in this repository. The course explicitly 
> asks that the dataset not be redistributed.

## Tech Stack

Python, PyTorch, HuggingFace Transformers (BERT, RoBERTa), NumPy, 
Matplotlib