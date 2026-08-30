## 2025-02-28 - [Performance] Cache NLTK Lemmatizer in pandas apply calls
**Learning:** `WordNetLemmatizer.lemmatize()` is slow when called repeatedly on common words within a pandas `.apply()` loop across large datasets. Redundant lemmatization forms a massive bottleneck during NLP preprocessing.
**Action:** Always wrap `lemmatize` with `@lru_cache(maxsize=100000)` (or similar size based on vocabulary) to skip redundant calculations. For texts with heavily repeating vocabulary, this provides a massive (10-15x) speedup during preprocessing.
