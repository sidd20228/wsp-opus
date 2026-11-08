```python
w=W['ln1_w.0']; b=W['ln1_b.0']
print(np.histogram(w, bins=20))
print(np.sort(w)[:20]); print(np.sort(w)[-40:])
print(np.histogram(W['ln2_w.0'], bins=20))
print(np.corrcoef(W['ln1_w.0'], W['ln2_w.0']))
```