## 2024-05-24 - [NLP Preprocessing Performance Bottlenecks]
**Learning:** [NLTK's `WordNetLemmatizer` is a known bottleneck for large datasets. Compiling regular expressions using `re.compile` also saves processing time when used repetitively.]
**Action:** [Always wrap `lemmatize()` calls with `functools.lru_cache` and pre-compile regular expressions (using `re.compile`) to achieve significant performance improvements in NLP preprocessing logic.]
