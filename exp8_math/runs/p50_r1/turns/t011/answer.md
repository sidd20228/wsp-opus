```python
# Hypothesis: same as before; inspect sample record structure and print excerpts of a cut-off generation.
s0 = G[0]['samples'][0]
print({k: (str(v)[:80]) for k, v in s0.items() if k != 'text'})
def asc(x): return x.encode('ascii', 'replace').decode()
t = s0['text']
print(len(t))
print('BEGIN:', asc(t[:600]))
print('...MID:', asc(t[len(t)//2: len(t)//2+600]))
print('...END:', asc(t[-600:]))
```