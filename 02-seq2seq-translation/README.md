# Experiment 2: seq2seq Neural Machine Translation

## Overview

I built a Chinese-English neural machine translation system based on the 
seq2seq framework, starting from a basic RNN encoder-decoder and improving 
it step by step. My work covers:

- Modifying the provided skeleton (special token indices, low-frequency 
  filtering, validation split).
- Inference tuning: BeamSearch with repetition and length penalties.
- Training tuning: hyperparameter analysis and adjustment.
- Structural improvements: GRU and Attention (bidirectional GRU encoder + 
  attention in the decoder).
- Batched training for better GPU utilization.
- Final test-set inference with the best configuration.

## What I Did

### Base model
- Modified the provided skeleton:
  - Adjusted special token indices: `<SOS>=1`, `<EOS>=2`, reserved `0` 
    for padding, added `<UNK>=3`.
  - Added a low-frequency word filter (`min_count=2`).
  - Split 10% of the training data as a validation set for BLEU 
    computation.

### Inference tuning
- Implemented BeamSearch with two constraints:
  - Repetition penalty (to prevent repeated output).
  - Length penalty (to encourage outputs of reasonable length).

### Training tuning
- Analyzed the original hyperparameters (`hidden_size=500`, `n_layers=2`, 
  `dropout_p=0.05`, `learning_rate=0.0001`).
- Tried increasing `dropout_p` to 0.2 and `learning_rate` to 0.0005, and 
  lowering `teacher_forcing_ratio` to 0.3.

### Structural improvements
- **GRU**: Replaced RNN with GRU in both encoder and decoder.
- **Attention**: Built on the GRU model with a bidirectional GRU encoder 
  and an attention mechanism in the decoder. `hidden_size` reduced to 256 
  to speed up training.

### Batched training
- Rewrote the training loop to support batched processing, improving GPU 
  utilization.
- Tried two hyperparameter sets:
  - (A) `hidden_size=256`, `n_layers=2`, `dropout=0.1`, `lr=0.001`, 
    `batch_size=32`, `n_epochs=20`.
  - (B) `hidden_size=512`, `n_layers=2`, `dropout=0.2`, `lr=0.001`, 
    `batch_size=32`, `n_epochs=10`.
- Chose (A) for final test-set inference, and applied BeamSearch on top 
  of it.

## Key Results

| Model | Corpus BLEU | Avg Sentence BLEU |
|-------|-------------|-------------------|
| Base RNN | 0.1519 | 0.1411 |
| Base + BeamSearch | 0.1263 | 0.1188 |
| Base + hyperparameter tuning | 0.1174 | 0.1058 |
| GRU | 0.3231 | 0.3140 |
| Attention (BiGRU + attention) | 0.4123 | 0.3983 |
| Batched Attention (A) | 0.7496 | 0.7137 |
| Batched Attention (B) | 0.6840 | 0.6450 |
| Batched Attention (A) + BeamSearch | **0.7684** | **0.7362** |

Final best: **Corpus BLEU 0.7684** (batched Attention + BeamSearch).

## Analysis

- **Base RNN**: only translates a few high-frequency words and generates 
  many trailing periods. Corpus BLEU 0.1519.
- **BeamSearch**: repetition and overly short outputs were reduced, but 
  BLEU dropped to 0.1263 — the decoder's underlying ability was the 
  bottleneck, not the search strategy.
- **Hyperparameter tuning**: raising `learning_rate` caused the loss to 
  converge too early; BLEU dropped to 0.1174.
- **Diagnosis**: since neither inference tuning nor training tuning helped, 
  I concluded the RNN structure itself was the bottleneck and shifted to 
  structural improvements.
- **GRU**: BLEU jumped to 0.3231 — replacing RNN with gated units clearly 
  helped.
- **Attention**: BLEU reached 0.4123 (~27% over GRU). Loss curve smoothness 
  ranked Attention > GRU > RNN.
- **Batched training**: made training much more efficient and let the model 
  converge properly. Batched Attention (A) reached BLEU 0.7496 — a large 
  jump over the non-batched Attention model (0.4123).
- **BeamSearch on top of (A)**: BLEU 0.7684, slightly better than greedy.

## Files

- `README.md` — this file (English)
- `README_zh.md` — Chinese version
- `report.pdf` — full experiment report (PDF)
- `seq2seq-translation_base.ipynb` — base RNN model 
  (modified from the provided skeleton)
- `seq2seq-translation_gru.ipynb` — GRU model
- `seq2seq-translation_attention.ipynb` — Attention + batched training
- `test_translations_batch.txt` — source sentences and model translations 
  on the test set (batched Attention model)

## Datasets (not included)

- **NIST Chinese-English parallel corpus** — 90,000 sentence pairs, 
  trimmed to 68,898 pairs.
  - Train: 62,009 (90%)
  - Validation: 6,889 (10%)
  - Input vocabulary: 3,883
  - Output vocabulary: 8,855

> Full dataset is not included in this repository. Only the report, 
> notebooks, and the final test-set translations are provided for 
> demonstration purposes.

## Tech Stack

Python, PyTorch, NLTK (BLEU), Matplotlib