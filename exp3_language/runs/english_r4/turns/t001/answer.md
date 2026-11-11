```python
for k,v in W.items():
    print(k, v.shape, float(np.abs(v).mean()), float(v.std()))
```