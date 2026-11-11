loop_info and loop_stats2 take token ids, not text. I'll tokenize one cut sample and call both to see what they return: period, onset and run length. Then I'll tokenize all 3×512 samples once, cache the ids, and compute the finish and loop-onset hazard curves.

```python
# Check: loop_info/loop_stats2 outputs on token ids of one cut sample
ids0 = tok.encode(F11[k0]['text']).ids
print(len(ids0), F11[k0]['tokens'])
print("loop_info:", loop_info(ids0)); r2 = loop_stats2(ids0); print("loop_stats2:", r2 if not isinstance(r2,dict) else {a:b for a,b in r2.items() if not isinstance(b,(list,np.ndarray))})
```