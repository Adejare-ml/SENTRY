## 2024-05-24 - [NLTK Lemmatizer Performance]
**Learning:** NLTK's `WordNetLemmatizer.lemmatize()` is very slow when called repeatedly for every word in large datasets (like 447k+ rows).
**Action:** Always wrap `lemmatize()` with `functools.lru_cache(maxsize=50000)` to cache results for common words. Combine this with pre-compiling regular expressions for text cleaning to achieve a ~10-20x speedup in NLP preprocessing pipelines.
