File references aren't supported, so texts must be pasted inline, and teacher-forced inputs have to stay compact. To keep the attention pass reliable to transcribe, I'll use two looping transcripts with an early onset and truncate each to about 1,150 tokens. That gives ~600–700 tokens of normal reasoning plus 15+ loop cycles, enough to measure per-head attention to the previous cycle. Candidates: math_train_geometry_151 sample 0 (loop at ~708 absolute tokens, period 30) and gsm8k_train_2598 sample 0 (~805, period 17). First I'll check that the truncation lands well inside the loop and see how long the JSON-escaped strings are.

```python
# Hypothesis: (setup) shorter looping texts (~1150 tok) suffice for induction-head identification; check them.
TA, idsA = build_text('math_train_geometry_151', 0, 1150)
TB, idsB = build_text('gsm8k_train_2598', 0, 1150)
for T, ids in [(TA, idsA), (TB, idsB)]:
    js = json.dumps(T)
    print(len(ids), len(tok.encode(T).ids), len(js), 'nonascii:', sum(ord(ch) > 127 for ch in T))
    print(asc(repr(T[-250:])))
```