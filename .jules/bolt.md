## 2024-06-15 - [Pre-compiled regex for apply functions]
**Learning:** Compiling regex on the fly inside a function mapped over hundreds of thousands of rows via Pandas `.apply()` acts as a significant bottleneck.
**Action:** When using `.apply()` with regex logic, pre-compile the regex `re.compile()` outside the function to avoid redundant compilations.
