```python
svd=np.linalg.svd
s=svd(W['W_proj.3'],compute_uv=False)
print('W_proj.3 sv',s[:60].round(2))
Wo=W['W_o.3']
print('W_o.3 sv',svd(Wo,compute_uv=False)[:60].round(2))
for l in range(3):
    print(f'W_proj.{l} sv',svd(W[f'W_proj.{l}'],compute_uv=False)[:50].round(2))
```