Each record holds one problem and a list of samples. I need the sample fields (grade, length, cut flag) before I can pair across passes. I'll print the sample keys and one compact sample without the text, then build the paired table keyed by problem id and sample index.

```python
# Inspect sample structure, then build paired correctness by (id, sample idx) across gen_8/gen_9/gen_12.
s0 = g12[0]['samples'][0]
print(sorted(s0.keys())); print({k:(v if not isinstance(v,str) else v[:80]) for k,v in s0.items() if k!='text'})
```