## 2024-09-01 - [Python/Pandas regex optimization]
**Learning:** In pandas `.apply()` operations that process hundreds of thousands of rows of text, compiling a regex pattern inside the applied function using `re.sub(r'pattern', ...)` can cause significant performance overhead because the regex is recompiled or looked up in cache on every single function call.
**Action:** Always pre-compile regex patterns using `re.compile()` outside of functions that are applied across large pandas DataFrames to prevent repeated recompilation overhead.
