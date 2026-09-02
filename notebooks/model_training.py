#!/usr/bin/env python
# coding: utf-8

# # SENTRY — ML Model Training
#
# Trains two fraud/spam detection models used by the SENTRY platform:
#
# 1. **Fraud Detector** — Random Forest on the Enron email dataset (`data/enron_data_fraud_labeled.csv`)
# 2. **Spam Classifier** — Random Forest on the SMS Spam Collection (`data/spam.csv`)
#
# **Pipeline design:** TF-IDF vectorizer is fit **only on training data** to prevent data leakage.
# Train/test split: 80/20, stratified by label.
#
# **Outputs saved to `models/`:**
# - `enron_fraud_rf_model.pkl` + `tfidf_vectorizer.pkl`
# - `spam_detector_rf.pkl` + `spam_tfidf_vectorizer.pkl`
#

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
import joblib
import nltk
from pathlib import Path
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_auc_score, accuracy_score
)

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Paths — cross-platform, relative to notebook
DATA_DIR   = Path('data')
MODELS_DIR = Path('models')
MODELS_DIR.mkdir(exist_ok=True)


# In[23]:


# Data loading — Enron email fraud dataset
df = pd.read_csv(DATA_DIR / 'enron_data_fraud_labeled.csv', on_bad_lines='skip', low_memory=False)
print(df.shape)
df.info()


# In[24]:


# Preprocessing
df_clean = df[['Body', 'Label']].dropna()

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

import functools

compiled_re = re.compile(r'[^a-zA-Z\s]')

@functools.lru_cache(maxsize=50000)
def cached_lemmatize(w):
    return lemmatizer.lemmatize(w)

def clean_text(text: str) -> str:
    """Lowercase, strip non-alpha, remove stopwords, lemmatize."""
    # Bolt Optimization: Pre-compiled regex and LRU cache for lemmatization
    # to significantly speed up processing of 447k+ rows.
    text = compiled_re.sub('', str(text).lower())
    words = [cached_lemmatize(w) for w in text.split() if w not in stop_words]
    return ' '.join(words)

# Note: applying to large datasets (447k+ rows) may take 5–10 min
df_clean['Body'] = df_clean['Body'].apply(clean_text)
print(f'Cleaned {len(df_clean):,} rows')


# In[25]:


# Train/test split — stratified to preserve class balance
X_train, X_test, y_train, y_test = train_test_split(
    df_clean['Body'], df_clean['Label'],
    test_size=0.2, random_state=42, stratify=df_clean['Label']
)

# TF-IDF: fit ONLY on training data to prevent data leakage
tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X_train_tfidf = tfidf.fit_transform(X_train)   # fit + transform train
X_test_tfidf  = tfidf.transform(X_test)         # transform only (no fit)

print(f'Train: {X_train_tfidf.shape}, Test: {X_test_tfidf.shape}')

# Train
rf_model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
print('Training...')
rf_model.fit(X_train_tfidf, y_train)
print('Done.')


# In[26]:


# Evaluation
y_pred = rf_model.predict(X_test_tfidf)
y_prob = rf_model.predict_proba(X_test_tfidf)[:, 1]

print(f'Accuracy : {accuracy_score(y_test, y_pred):.4f}')
print(f'ROC-AUC  : {roc_auc_score(y_test, y_prob):.4f}')
print()
print(classification_report(y_test, y_pred, target_names=['Legit', 'Fraud']))

plt.figure(figsize=(6, 4))
sns.heatmap(
    confusion_matrix(y_test, y_pred),
    annot=True, fmt='d', cmap='Blues',
    xticklabels=['Legit', 'Fraud'],
    yticklabels=['Legit', 'Fraud']
)
plt.title('Confusion Matrix — Enron Fraud Detector')
plt.ylabel('Actual'); plt.xlabel('Predicted')
plt.tight_layout()
plt.show()


# In[27]:


# Save model and vectorizer
joblib.dump(rf_model, MODELS_DIR / 'enron_fraud_rf_model.pkl')
joblib.dump(tfidf,    MODELS_DIR / 'tfidf_vectorizer.pkl')
print('Fraud model saved to models/')


# ## Spam Detection Model Training
#

# In[30]:


# All imports already loaded above — skipping duplicates
# Using same clean_text(), stop_words, lemmatizer from Part 1


# In[32]:


# Data loading — SMS Spam Collection
df_spam = pd.read_csv(DATA_DIR / 'spam.csv', encoding='latin-1')
df_spam = df_spam[['v1', 'v2']]
df_spam.columns = ['Label', 'Body']
df_spam['Label'] = df_spam['Label'].map({'ham': 0, 'spam': 1})

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

df_spam['Body'] = df_spam['Body'].apply(clean_text)

# Train/test split — stratified
X_tr, X_te, y_tr, y_te = train_test_split(
    df_spam['Body'], df_spam['Label'],
    test_size=0.2, random_state=42, stratify=df_spam['Label']
)

# TF-IDF fit on train only
spam_tfidf = TfidfVectorizer(max_features=5000)
X_tr_tfidf = spam_tfidf.fit_transform(X_tr)
X_te_tfidf = spam_tfidf.transform(X_te)

# Train
spam_rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
spam_rf.fit(X_tr_tfidf, y_tr)

# Evaluate
sp_pred = spam_rf.predict(X_te_tfidf)
sp_prob = spam_rf.predict_proba(X_te_tfidf)[:, 1]
print(f'Accuracy : {accuracy_score(y_te, sp_pred):.4f}')
print(f'ROC-AUC  : {roc_auc_score(y_te, sp_prob):.4f}')
print(classification_report(y_te, sp_pred, target_names=['Ham', 'Spam']))

# Save
joblib.dump(spam_rf,    MODELS_DIR / 'spam_detector_rf.pkl')
joblib.dump(spam_tfidf, MODELS_DIR / 'spam_tfidf_vectorizer.pkl')
print('Spam model saved to models/')
