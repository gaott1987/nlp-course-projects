# Experiment 1: Word2Vec & TransE

## Overview

This experiment covers two classic representation learning models: 
Word2Vec for word embeddings and TransE for knowledge graph embeddings. 
I chose to focus not just on getting the models to run, but on 
**understanding how hyperparameters affect embedding quality** and 
**diagnosing why a trained model can still fail at simple inference tasks**.

For Word2Vec, I merged the provided training and evaluation scripts into 
a single pipeline and ran three parameter sweeps to see how vector size, 
window, and min_count affect Spearman correlation on WordSim353.

For TransE, after completing the two missing functions in the provided 
skeleton, I went beyond the assignment by building my own inference and 
diagnostic tools: I implemented task 1 (head + relation → tail) and 
task 2 (head + tail → relation), analyzed embedding norm distributions, 
and found that the trained vectors had abnormally small norms. I then 
manually L2-normalized the vectors at inference time and showed that this 
significantly improved the ranking of the correct answer.

## What I Did

### Word2Vec
- Ran the provided `word2vec.py` and `evaluate.py` to get the baseline 
  Spearman correlation on WordSim353 (~0.69).
- Implemented `main_function(vector_size, window, min_count)` and 
  `plot_spcors(x, y, xlabel)` in the notebook to merge training and 
  evaluation into one pipeline.
- Ran three parameter sweeps:
  - `vector_size` from 50 to 500 (step 50)
  - `window` from 2 to 20 (step 2)
  - `min_count` from 2 to 20 (step 2) and 20 to 100 (step 10)

### TransE
- Completed `_calc()` and `loss()` in the provided `TransE.py` skeleton.
- Trained for 500 epochs.
- Implemented `load_id_mapping()` and `load_vectors()` to load ID mappings 
  and embedding files.
- Implemented `task1_find_closest_tails()` and 
  `task2_find_closest_relations()` for inference and verification.
- Implemented `analyze_embeddings_quality()` to inspect embedding norms 
  and value distributions.
- Implemented `task1_with_normalization()` to manually L2-normalize 
  vectors at inference time and re-run task 1.

## Key Results

### Word2Vec (Spearman correlation on WordSim353)

| Setting | Spearman |
|---------|----------|
| Baseline (vector_size=200, window=10, min_count=10) | ~0.69 |
| Best (vector_size=400) | 0.699 |

- `vector_size` 100–200 is a reasonable range; further increases give no 
  clear improvement.
- `window` 10–15 is reasonable; below 10 the correlation drops sharply.
- `min_count` in 20–100 shows no clear correlation with performance.

### TransE

- Trained for 500 epochs; loss dropped from 91 to 0.46.
- Task 1 (Q30 + P36 → tail): expected Q61 (Washington, D.C.) did not 
  appear in top 10; the top result was Q30 itself.
- Task 2 (Q30 + Q49 → relation): expected P30 (continent) did not appear 
  in top 5.
- After manual L2 normalization at inference: **Q61 ranking improved 
  to 85**.

## Analysis

- **Word2Vec**: The Spearman correlation saturates quickly. Increasing 
  vector size beyond ~100–200 does not help, which suggests the corpus 
  (Text8) and evaluation set (WordSim353) reach a ceiling for this 
  model family. Window size matters more: too small a window misses 
  long-range semantic relations; 10–15 is a good trade-off.
- **TransE**: The loss went down, but the inference results were 
  clearly wrong. My diagnostic (`analyze_embeddings_quality`) showed 
  that entity embeddings had a mean L2 norm of only ~0.046, which is 
  abnormally small. This means the model was not enforcing the 
  normalization constraint that TransE assumes (`||h|| = ||r|| = ||t|| = 1`).
  I did not retrain, but manually normalized the vectors at inference 
  and re-ran task 1. The correct answer Q61 went from unranked to 
  rank 85, confirming that the problem was normalization, not the 
  model's learned structure.

## Files

- `README.md` — this file (English)
- `README_zh.md` — Chinese version
- `report.ipynb` — full experiment report (Word2Vec + TransE)
- `report.pdf` — PDF version of the report

> All code (Word2Vec parameter analysis, TransE `_calc()`/`loss()` 
> completion, task 1/2 inference, embedding quality analysis, and 
> normalization experiment) is implemented inside `report.ipynb`. 
> No separate `.py` files are included.

## Datasets (not included)

- **Text8** — used for Word2Vec training. ~100 MB of cleaned Wikipedia text.
- **WordSim353** — word similarity benchmark; 353 word pairs with human 
  similarity scores, used to evaluate Word2Vec via Spearman correlation.
- **Wikidata** — used for TransE training. Contains 50,000 entities and 
  378 relations (as used in this experiment). Each triple is 
  (head, relation, tail).

> Full datasets and trained vectors are not included in this repository.

## Tech Stack

Python, Gensim (Word2Vec), NumPy, SciPy (Spearman), Matplotlib, 
PyTorch (TransE)