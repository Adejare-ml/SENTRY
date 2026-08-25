## 2024-05-30 - Caching NLP Lemmatization
**Learning:** Using `lru_cache` or a dictionary to cache NLP operations like `lemmatizer.lemmatize()` provides a massive performance boost (from ~12.6s to ~0.6s for 1000 items in a dummy benchmark) because natural language text has a high repetition of words.
**Action:** Apply caching (e.g. `functools.lru_cache` or a simple memoization dictionary) to functions that perform expensive word-by-word operations like lemmatization, especially in NLP data pipelines.
