## 2024-05-18 - [NLTK WordNetLemmatizer Performance]
**Learning:** NLTK's `WordNetLemmatizer.lemmatize` function is a significant bottleneck when processing large datasets because it performs expensive morphological analysis on each call. Pre-compiling regex also avoids compiling the regex inside a loop/map.
**Action:** Wrap `lemmatize()` calls with `functools.lru_cache` and pre-compile regular expressions (using `re.compile`) to achieve massive speedups on NLP preprocessing tasks.
