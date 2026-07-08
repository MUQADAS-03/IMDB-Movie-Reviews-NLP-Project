# IMDB Sentiment Analysis: Classical NLP vs. Neural Networks

## Project Overview
The project compares two fundamentally different approaches to sentiment analysis on the **IMDB movie reviews dataset**: classical NLP pipelines (TF-IDF + traditional ML classifiers) versus a neural network approach (SimpleRNN with word embeddings). The goal was to test a common assumption in NLP — that deep learning models automatically outperform classical methods — and evaluate it directly against a real dataset.

## Key Findings
| Approach | Accuracy | Notes |
|---|---|---|
| TF-IDF + Logistic Regression / SVM | **88–89%** | Fast, interpretable, production-ready |
| SimpleRNN (Word2Vec embeddings) | **~50%** | Struggled due to embedding and architecture challenges |

**Core insight:** the classical pipeline outperformed the neural network by a wide margin. Word2Vec embeddings captured semantic meaning well individually, but averaging them for the RNN input lost word-order information a critical signal for sentiment, where phrasing and negation (e.g. "not good" vs. "good") matter. This reinforced that model complexity isn't a substitute for correct architecture and feature representation, especially when data or tuning resources are limited.

## Approaches Compared

**1. Classical NLP Pipeline**
- Text preprocessing (cleaning, tokenization, stopword removal)
- TF-IDF vectorization to convert text into weighted numeric features
- Logistic Regression and SVM classifiers trained on the TF-IDF features

**2. Neural Network Pipeline**
- Word2Vec embeddings to represent words as dense semantic vectors
- SimpleRNN architecture to process review sequences
- Embeddings averaged before feeding into the network, which lost sequential/order information

## Dataset Information
| Attribute | Detail |
|---|---|
| Name | IMDB Movie Reviews Dataset |
| Task | Binary sentiment classification (positive / negative) |
| Content | Movie review text labeled by sentiment |

## Environment Setup
```bash
git clone https://github.com/MUQADAS-03/imdb-sentiment-analysis.git
cd imdb-sentiment-analysis
pip install -r requirements.txt
```

## Conclusion
This project's core takeaway goes against a common default assumption in applied ML: bigger/more complex models are not automatically better. A carefully engineered classical NLP pipeline (TF-IDF + Logistic Regression/SVM) beat a neural network by roughly 38 percentage points on this task, while also being faster to train, easier to interpret, and simpler to deploy in production. The neural network's underperformance traces back to a specific, identifiable cause — loss of word order from averaging embeddings — rather than the architecture being inherently unsuitable, pointing to sequence-preserving alternatives (e.g. LSTM/GRU without averaging, or attention-based pooling) as the natural next step if the neural approach were to be revisited.

## Author
**Muqadas Yasin**
