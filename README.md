# 📚 Kindle Review Sentiment Analysis using Word2Vec + Random Forest

A binary sentiment classifier for Amazon Kindle Store book reviews. Review text is cleaned, converted into 200-dimensional **Word2Vec** embeddings (averaged per review), and classified as **positive** or **negative** using a **Random Forest**.

---

## 📌 Overview

**Goal:** Predict whether a Kindle book review expresses positive or negative sentiment from its text alone.

**Pipeline:**

```
Raw review → Clean text → Tokenize → Word2Vec (200-d) → Average word vectors → Random Forest → Positive / Negative
```

---

## 📂 Dataset

- **Source:** [Amazon Kindle Store reviews (5-core)](http://jmcauley.ucsd.edu/data/amazon/), compiled by Julian McAuley, UCSD (also available on Kaggle as `all_kindle_review.csv`)
- **Subset used:** 12,000 reviews
- **Columns used:** `reviewText` (input) and `rating` (target)

**Label definition:**

| Rating | Sentiment | Label |
|---|---|---|
| 4 – 5 | Positive | `1` |
| 1 – 3 | Negative | `0` |

The resulting dataset is **perfectly balanced**: 6,000 positive and 6,000 negative reviews, so no resampling was needed. There are no missing values in the columns used.

---

## 🛠️ Tech Stack

| Area | Tools |
|---|---|
| Language | Python 3 |
| NLP | NLTK (stopwords, tokenizer), Gensim (Word2Vec), BeautifulSoup, `re` |
| ML | scikit-learn (RandomForestClassifier, metrics) |
| Data | Pandas, NumPy |
| Utilities | tqdm |

---

## 🔬 Methodology

1. **Label creation:** Converted 1–5 star ratings to binary sentiment (rating > 3 → positive).
2. **Text cleaning:**
   - Lowercasing
   - Removing special characters
   - Removing English stopwords (NLTK)
   - Removing URLs and HTML tags
   - Collapsing extra whitespace
3. **Tokenization:** Sentence tokenization (NLTK) followed by `gensim.utils.simple_preprocess`.
4. **Embeddings:** Trained a **Word2Vec** model on the review corpus (`vector_size=200`, `epochs=10`). The pretrained `word2vec-google-news-300` model was also loaded for exploration.
5. **Review vectors:** Each review is represented as the **mean of its word vectors** (zero vector if no known words), giving a feature matrix of shape `(12000, 200)`.
6. **Split:** 75% train / 25% test (`random_state=42`).
7. **Model:** `RandomForestClassifier(n_estimators=200, criterion='gini')`.
8. **Inference helper:** A `preprocess_and_embed()` function applies the same cleaning and embedding steps to new, unseen reviews for prediction.

---

## 📊 Results

Evaluated on 3,000 held-out reviews:

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| Negative (0) | 0.78 | 0.79 | 0.79 | 1501 |
| Positive (1) | 0.79 | 0.78 | 0.79 | 1499 |

- **Accuracy: 78.7%**
- **Confusion matrix:**

|  | Pred. Negative | Pred. Positive |
|---|---|---|
| **Actual Negative** | 1188 | 313 |
| **Actual Positive** | 327 | 1172 |

Performance is symmetric across both classes, consistent with the balanced dataset.

### Example predictions

| Review | Predicted |
|---|---|
| "This book was absolutely amazing! I loved every single page and couldn't put it down." | Positive |
| "Terrible product, completely useless and a waste of money." | Negative |

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 2. Install dependencies

```bash
pip install pandas numpy nltk gensim scikit-learn beautifulsoup4 tqdm jupyter
```

### 3. Add the dataset

Download the Kindle reviews CSV and place it in the project folder as `all_kindle_review .csv`. Update the path in the notebook if you are not using Google Colab:

```python
data = pd.read_csv('all_kindle_review .csv', index_col=0)
```

### 4. Download NLTK resources

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt_tab')
```

### 5. Run the notebook

```bash
jupyter notebook Kindle_Sentiment_Analysis.ipynb
```

> **Note:** The pretrained Google News Word2Vec model is ~1.6 GB and downloads on first use.

---

## 📁 Project Structure

```
├── Kindle_Sentiment_Analysis.ipynb   # Full workflow
├── all_kindle_review .csv            # Dataset (add manually)
└── README.md
```

---

## 🔭 Future Improvements

- Use the **pretrained Google News vectors** (or GloVe / FastText) for the review embeddings instead of training Word2Vec on 12k reviews.
- Replace mean-pooled vectors with **TF-IDF-weighted averaging**, or move to a sequence model (LSTM / BiLSTM) or a fine-tuned transformer such as DistilBERT.
- Establish baselines with **TF-IDF + Logistic Regression / Naive Bayes**, which often match or beat averaged embeddings on review sentiment.
- Add **hyperparameter tuning** (`GridSearchCV` / `RandomizedSearchCV`) and cross-validation.
- Train Word2Vec on the **training split only** to avoid leaking test-set text into the embeddings.
- Treat 3-star reviews as neutral (three-class problem) instead of grouping them with negatives.
- Deploy as a small web app (Streamlit / Flask) for live review scoring.

---

## 🙋 Author

**Ekangsh**
B.Tech, Production & Industrial Engineering, NIT Jamshedpur
Interested in data science and analytics.

Feel free to ⭐ the repo if you found it useful!