I'll use gsm8k_train_7170 (sample 0: loop at 6,764 tokens, period 42) and math_train_precalculus_281 (sample 0: loop at 4,571, period 43). Both are typical mid-length loops from different sources. Now I'll build the spliced texts with the exact special-token prompt, confirm the token counts are ≤2,048, check for non-ASCII characters (which I'd need to escape), and see where the loop starts within each text.

```python
# Hypothesis: (setup) build final spliced texts; check token length, ascii, and loop position inside the window.
TA, offA = splice_text('gsm8k_train_7170', 0, 850, 950)
TB, offB = splice_text('math_train_precalculus_281', 0, 850, 950)
for T, off in [(TA, offA), (TB, offB)]:
    ids = tok.encode(T).ids
    print(len(ids), 'nonascii:', sum(ord(ch) > 127 for ch in T), 'json len', len(json.dumps(T)), 'loop starts ~tok', len(tok.encode(T).ids) - 950)
    print(asc(repr(T[:120])), '...', asc(repr(T[-200:])))
```