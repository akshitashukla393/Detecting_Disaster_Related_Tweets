# Disaster Tweet Classification Project

## Overview

This project addresses the challenge of classifying tweets related to disasters. Social media, especially Twitter, is a vital source of real-time information during disasters. However, not every tweet mentioning disaster-related keywords actually pertains to a real disaster event. This project aims to automatically classify tweets into three categories:

- **Class 0**: Disaster word used but no actual disaster occurred  
- **Class 1**: Disaster word used and disaster occurred  
- **Class 2**: Not disaster related

---

## Problem Statement

Many tweets contain disaster-related keywords but do not necessarily indicate an ongoing disaster event. Distinguishing between genuine disaster-related tweets and irrelevant or sarcastic ones can improve disaster response and situational awareness. This project tackles this classification problem to assist emergency responders, news agencies, and the public in filtering valuable information quickly and accurately.

---

## Background & Literature Review

- Previous studies often focus on keyword matching or sentiment analysis but struggle with sarcasm and context detection.
- Feature extraction techniques explored:
  - TF-IDF, word bi-grams, and tri-grams
  - Word embeddings like **GloVe** and **Word2Vec**
- Classification models evaluated:
  - Logistic Regression
  - Random Forest
  - Gradient Boosting (Best performing)
  - XGBoost
- **Gradient Boosting** trained on a combination of TF-IDF features and embedding-based sentiment features performed best.

---

## Data

- **Training Data**: `combinedSarcasmDataset.csv` (7,563 samples)  
- **Test Data**: `disasterTest.csv` (3,263 samples)  
- **Sentiment Data**: `sentimentAnalysis.csv` (40,000 samples)  

### Label Distribution in Training Data

| Class | Description                          | Samples | Percentage |
|-------|--------------------------------------|---------|------------|
| 0     | Disaster word but no disaster        | 3009    | 39.79%     |
| 1     | Disaster word and disaster occurred  | 1883    | 24.90%     |
| 2     | Not disaster related                 | 2671    | 35.32%     |

---

## Methodology

### Step 1: Data Preprocessing
- Text cleaning and normalization  
- Extraction of rich text features (e.g., length, punctuation, capitalization)  
- Sentiment extraction using GloVe embeddings and a sentiment analysis model

### Step 2: Feature Engineering
- TF-IDF vectors with unigrams, bigrams, and trigrams  
- Sentiment features via word embeddings  
- Combination of categorical and rich text features

### Step 3: Model Training & Evaluation
- Cross-validation on:
  - Random Forest
  - Gradient Boosting (best performer)
  - XGBoost
- **Gradient Boosting** achieved the highest performance (~70% accuracy and F1-score)
- Final model saved as `.pkl` file for inference

---

## Results

- **Accuracy**: 70.05% ± 1.29%  
- **Precision**: 70.72% ± 1.32%  
- **Recall**: 70.05% ± 1.29%  
- **F1 Macro**: 70.84% ± 1.27%  

### Prediction Distribution on Test Data
- Class 0: 45.72%  
- Class 1: 21.48%  
- Class 2: 32.79%  

Sample prediction outputs include:
- Probabilities for each class
- Final predicted class based on highest probability

---

## How to Use

### Running the Dashboard

A **Streamlit** dashboard is available to interactively test tweet classification.

```bash
python -m streamlit run dashboardFifteen.py
```

1. Upload the trained `.pkl` model file through the dashboard.
2. Enter a tweet for classification.
3. The dashboard returns:
   - Probability scores for each class
   - The predicted class label

---

## Code Structure

- **Data Loading & Preprocessing**: Scripts to load datasets, clean text, extract features  
- **Embedding Models**: Load GloVe and Word2Vec embeddings  
- **Model Training**: Train and cross-validate models  
- **Model Saving**: Save best model as `.pkl`  
- **Prediction Scripts**: Predict class for new tweets using saved model  
- **Dashboard**: `dashboardFifteen.py` runs the Streamlit app for prediction

---

## Future Work

- Improve sentiment analysis accuracy  
- Use transformer-based models (e.g., **BERT**) for better context  
- Expand dataset, especially for sarcasm detection  
- Enhance dashboard UI/UX for easier use

---

## Acknowledgements

- Datasets from **Kaggle**  
- **GloVe** and **Word2Vec** embedding providers  
- Open-source libraries:
  - `scikit-learn`
  - `XGBoost`
  - `Streamlit`
  - `NLTK`
  - `pandas`
  - `numpy`
