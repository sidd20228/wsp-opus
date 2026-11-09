```python
SHAPES_=SHAPES
exec(open('/work/build.py').read())
W=build(P0)
for n in NAMES: save(n,W[n])
print('saved', W['L6.W'][:, :81].sum(), np.abs(W['L5.W']).sum())
```