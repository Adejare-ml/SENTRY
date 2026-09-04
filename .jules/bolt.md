## 2024-05-18 - [Optimize NLP preprocessing]
**Learning:** NLTK's `WordNetLemmatizer` is a significant performance bottleneck for large datasets in this project.
**Action:** Always wrap `lemmatize()` calls with `functools.lru_cache` and pre-compile regular expressions (using `re.compile`) when doing NLP text preprocessing to achieve major speedups.
