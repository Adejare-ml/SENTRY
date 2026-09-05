## 2024-05-18 - NLP Preprocessing Bottleneck
**Learning:** NLTK's `WordNetLemmatizer` and regex parsing within a loop are major bottlenecks when processing large datasets (like the 447k+ Enron emails) inside `apply()`.
**Action:** Always wrap `lemmatizer.lemmatize()` calls with `functools.lru_cache` and pre-compile regular expressions (using `re.compile`) when doing NLP preprocessing on large datasets in this codebase to achieve significant performance improvements.
