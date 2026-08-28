## 2024-05-24 - [NLTK Lemmatizer Bottleneck]
**Learning:** NLTK's `WordNetLemmatizer` is a major bottleneck on large datasets because it performs complex dictionary lookups internally for every word, even when words are repeated often in natural language.
**Action:** Wrapping it with `functools.lru_cache` yields a massive speedup in NLP preprocessing pipelines. Always cache repetitive lemmatization or stemming operations on large corpora.

## 2024-05-24 - [Regex Compilation]
**Learning:** Calling `re.sub` directly with a string pattern inside a loop for hundreds of thousands of rows causes redundant regex compilation overhead.
**Action:** Pre-compile regex patterns using `re.compile` outside the loop when they will be reused extensively across large datasets.
