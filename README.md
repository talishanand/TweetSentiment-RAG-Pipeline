# TweetSentiment-RAG-Pipeline

A RAG-based Twitter sentiment analysis project that uses Google Gemini embeddings, ChromaDB retrieval, and an XGBoost classifier to predict Negative, Neutral, and Positive sentiment from tweet text.[1]

## Overview

This project builds a sentiment classification pipeline around tweet embeddings generated with Gemini and stores dense vectors for retrieval-oriented processing with ChromaDB.[1] The notebook also includes tweet cleaning, word cloud visualization, train-test splitting, multi-class XGBoost training, confusion matrix analysis, and example real-time predictions.[1]

## Features

- Google Gemini embeddings for converting tweets into dense vector representations.[1]
- ChromaDB support for retrieval-style vector storage and similarity workflows.[1]
- XGBoost multi-class classifier for Negative, Neutral, and Positive sentiment prediction.[1]
- Text preprocessing with regex-based cleaning for URLs, mentions, hashtags, punctuation, and casing normalization.[1]
- Word cloud generation for quick exploratory text visualization.[1]
- Evaluation using a classification report and confusion matrix.[1]
- Example prediction function for real-time sentiment inference on new text.[1]

## Dataset

The notebook loads a `Tweets.csv` dataset and keeps the `sentiment` and `text` columns for modeling.[1] The discovered sentiment classes are Negative, Neutral, and Positive, and the notebook maps them to numeric labels 0, 1, and 2 for training.[1]

## Pipeline

1. Load the tweet dataset and keep the text and sentiment fields.[1]
2. Clean tweet text by removing URLs, mentions, hashtags, punctuation, and extra noise.[1]
3. Generate Gemini embeddings for cleaned tweets using the embedding API.[1]
4. Encode sentiment labels and split the data with stratified train-test sampling.[1]
5. Train an XGBoost classifier configured for 3-class sentiment prediction.[1]
6. Evaluate results with accuracy-oriented metrics, per-class precision/recall/F1, and a confusion matrix.[1]
7. Run example predictions on new tweets for real-time inference.[1]

## Tech Stack

| Component | Usage |
|---|---|
| Google Gemini Embeddings | Dense vector generation from tweet text.[1] |
| ChromaDB | Vector storage and retrieval-style workflows.[1] |
| XGBoost | Multi-class sentiment classification.[1] |
| scikit-learn | Train-test split and evaluation metrics.[1] |
| Pandas / NumPy | Data handling and numerical processing.[1] |
| Matplotlib / Seaborn / WordCloud | Visualization, confusion matrix, and word cloud output.[1] |

## Results

The notebook reports an example evaluation accuracy of 0.63 on the shown test output, along with class-wise precision, recall, and F1-scores for Negative, Neutral, and Positive labels.[1] It also demonstrates sample predictions such as “I loved it” being classified as Positive and “Pakistan cricket team is performing bad” being classified as Negative.[1]

## Setup

```bash
pip install google-generativeai xgboost chromadb umap-learn scikit-learn matplotlib seaborn pandas numpy wordcloud adjustText
```

Set your Google API key before running the notebook so Gemini embeddings can be generated.[1]

## Run

1. Open the notebook in Google Colab or Jupyter.[1]
2. Add your Google API key to the environment or notebook secrets.[1]
3. Place `Tweets.csv` in the expected path used by the notebook.[1]
4. Run the cells in order to preprocess data, generate embeddings, train the model, and evaluate predictions.[1]

## Repository Structure

```text
.
├── README.md
├── embed_2-2.ipynb
└── data/
    └── Tweets.csv
```

## Notes

The notebook currently uses the Gemini embedding workflow through `google.generativeai`, and the runtime output includes a deprecation warning recommending migration to the newer `google.genai` package in the future.[1] The notebook also appears to limit rows for demo speed in one step, so full-dataset runs may require removing that truncation line.[1]

## Future Improvements

- Migrate from `google.generativeai` to `google.genai`.[1]
- Add a dedicated retrieval step that uses nearest-neighbor examples directly during prediction.[1]
- Export the prediction pipeline as a small API or web app.
- Add experiment tracking and hyperparameter tuning for stronger model performance.
