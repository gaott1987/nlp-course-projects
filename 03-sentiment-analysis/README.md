# Experiment 3: Text Sentiment Analysis

## Overview

In this experiment I built a sentiment analysis pipeline on the Rotten Tomato 
movie review dataset. I implemented a feature extractor and a linear classifier 
trained with hinge loss from scratch, then extended the work by comparing six 
different text representation methods — from statistical features (BOW, N-gram, 
TF-IDF) to word/sentence embeddings (GloVe, Doc2Vec) and a contextual 
pre-trained model (BERT). I also replaced the hand-written linear classifier 
with an SVM to compare optimization behavior.

## What I Did

### Hinge loss derivation and implementation
- Derived the backpropagation of hinge loss with respect to the weight vector `w`.
- Implemented the linear classifier (`learnPredictor`) with SGD updates.

### Feature extraction
Implemented six text representation methods:
- **BOW** — 0-1 bag-of-words vector.
- **N-gram (N=2)** — bigram statistical features.
- **TF-IDF** — term frequency–inverse document frequency weighting.
- **Word2Vec (GloVe 100d)** — pre-trained GloVe vectors, sentence vector = average of word vectors.
- **Doc2Vec** — gensim Doc2Vec trained on the task data.
- **BERT** — [CLS] token representation as the sentence vector.

### Classifier comparison
- Trained the simple linear classifier with hinge loss and SGD on all features.
- Replaced the classifier with an SVM on the same features and compared ROC curves.

## Key Results

| Feature | Accuracy | F1 | AUC |
|---------|----------|-----|-----|
| BOW | 0.7364 | 0.7376 | 0.8080 |
| N-gram (N=2) | 0.6705 | 0.6741 | 0.7378 |
| TF-IDF | 0.7434 | 0.7396 | 0.8213 |
| Word2Vec (GloVe 100d, avg) | 0.7290 | 0.7133 | 0.8083 |
| Doc2Vec | 0.6497 | 0.6665 | 0.6946 |
| **BERT + SVM** | **0.7932** | **0.7879** | **0.8726** |

Best result: **BERT [CLS] + SVM, Accuracy 0.7932**.

## Analysis

- **BOW**: test set converged around epoch 10 while training loss kept 
  decreasing — clear overfitting.
- **N-gram**: converged faster than BOW but generalized worse. Sentiment is 
  often expressed by single words, and bigram negation patterns are too sparse 
  in this small dataset (3,554 reviews).
- **TF-IDF**: training error dropped to near-convergence in the first epoch, 
  and TF-IDF slightly outperformed BOW (0.7434 vs 0.7364).
- **Word2Vec (GloVe)**: train and test curves were very close, showing strong 
  generalization thanks to pre-trained vectors. However, overall metrics were 
  not clearly better than statistical features, and recall was relatively low.
- **Doc2Vec**: trained only on the small task dataset, so performance was the 
  worst (0.6497), even below statistical models.
- **SVM vs. linear classifier**: SVM did not outperform the simple linear 
  classifier, especially with GloVe features. The GloVe value distribution 
  does not match SVM's default optimizer settings, while SGD was more stable.
- **BERT + SVM**: clearly the best result (0.7932), confirming the advantage 
  of contextual pre-trained representations.

### Model comparison summary
| Model | Pre-training / Construction | Feature Nature | Performance |
|-------|------------------------------|----------------|-------------|
| BOW / N-gram / TF-IDF | No pre-training, statistical from training set | Word frequency, no semantics | Good |
| Word2Vec / GloVe | Static pre-training on word co-occurrence | Static word semantics, no context | Good |
| Doc2Vec | Self-trained on task data | Document-level topic vector, ignores local word order | Poor |
| BERT | Masked LM + NSP on large corpus | Deep contextual, dynamic per context | Very good |

## Files

- `README.md` — this file (English)
- `README_zh.md` — Chinese version
- `report.pdf` — full experiment report (PDF)
- `senti_analysis.ipynb` — all experiment code (base model + optional models)

## Datasets

- **Rotten Tomato movie review dataset** — 3,554 training reviews and 
  3,554 test reviews. Each review contains a sentiment label (+1 / -1) 
  and the review text.

> Full dataset is not included in this repository. Only the report and 
> the notebook are provided for demonstration purposes.

## Tech Stack

Python, NumPy, scikit-learn (TF-IDF, SVM, metrics), gensim (GloVe, Doc2Vec), 
transformers (BERT), Matplotlib