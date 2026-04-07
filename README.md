# IMDB Sentiment Analysis

A machine learning project to classify IMDB movie reviews as **positive** or **negative** using Natural Language Processing (NLP) techniques.

---

## Project Structure

```
sentiment-analysis/
│
├── IMDB Dataset.csv
│   
├── models/
│   ├── lr_model.pkl             # Trained Logistic Regression model
│   ├── tfidf_vectorizer.pkl     # TF-IDF vectorizer
│   └── label_encoder.pkl        # Label encoder
│
├── SentimentAnalysis.ipynb
│  
└── README.md
```

---

## Dataset

- **Source:** [IMDB Dataset on Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- **Size:** 50,000 movie reviews
- **Classes:** Positive / Negative (balanced — 25,000 each)
- **Columns:** `review`, `sentiment`

---

## Installation

```bash
pip install pandas numpy scikit-learn nltk matplotlib seaborn wordcloud beautifulsoup4
```

Download NLTK data:

```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')
```

---

## Pipeline

### 1. Preprocessing

| Step | Description |
|------|-------------|
| Lowercasing | Converts all text to lowercase |
| HTML removal | Removes `<br>` and other HTML tags |
| URL removal | Removes any web links |
| Punctuation removal | Strips all special characters |
| Digit removal | Removes numbers |
| Stopword removal | Removes common English stopwords (keeping negations) |
| Lemmatization | Reduces words to their base form |

```python
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_row(text):
    text = re.sub(r'<.*?>', '', text)           
    text = re.sub(r'https?://\S+|www\.\S+', '', text) 
    text = text.lower()                          
    text = re.sub(r'[^\w\s]', '', text)          
    text = re.sub(r'\d+', '', text)            
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return ' '.join(tokens)

df['review'] = df['review'].apply(preprocess_row)
```

### 2. Feature Extraction

TF-IDF Vectorization with bigrams:

```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), stop_words='english')
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf  = tfidf.transform(X_test)
```

### 3. Model Training

Three models were trained and compared:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

lr  = LogisticRegression(max_iter=1000)
nb  = MultinomialNB()
svm = LinearSVC()
```

---

## Results

| Model | Accuracy | F1 Score |
|-------|----------|----------|
| **Logistic Regression** | **88.88%** | **0.89** |
| SVM | 87.97% | 0.88 |
| Naive Bayes | 85.57% | 0.86 |

> **Best Model: Logistic Regression** with 88.88% accuracy and balanced precision/recall across both classes.

### Classification Report (Logistic Regression)

```
              precision    recall  f1-score   support

    Negative       0.90      0.87      0.89      4961
    Positive       0.88      0.90      0.89      5039

    accuracy                           0.89     10000
   macro avg       0.89      0.89      0.89     10000
weighted avg       0.89      0.89      0.89     10000
```

---

## Save & Load Model

### Save

```python
import pickle, os

os.makedirs('models', exist_ok=True)

with open('models/lr_model.pkl', 'wb') as f:
    pickle.dump(lr, f)

with open('models/tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

with open('models/label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)
```

### Load & Predict

```python
import pickle

with open('models/lr_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('models/tfidf_vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

with open('models/label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

def predict_sentiment(review):
    clean  = preprocess_row(review)
    vec    = tfidf.transform([clean])
    pred   = model.predict(vec)[0]
    prob   = model.predict_proba(vec)[0][pred]
    label  = le.inverse_transform([pred])[0]
    print(f"Sentiment : {label.upper()}")
    print(f"Confidence: {prob*100:.1f}%")

predict_sentiment("This movie was absolutely amazing!")
predict_sentiment("Terrible acting and a boring plot. Waste of time.")
```
---
## Future Improvements

- Fine-tune **BERT** for higher accuracy (target: 93%+)
- Build a **Streamlit web app** for live predictions

---

## Technologies Used

- **Python 3**
- **Pandas** — data manipulation
- **NLTK** — text preprocessing
- **Scikit-learn** — ML models and evaluation
- **Matplotlib / Seaborn** — visualization
- **WordCloud** — word cloud generation

