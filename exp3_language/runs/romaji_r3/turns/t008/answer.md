```python
np.set_printoptions(precision=2, suppress=True, linewidth=220)
for name in ['W_proj.3','W_o.3','W_proj.2','W_o.2','W_proj.1','W_proj.0']:
    s=np.linalg.svd(W[name],compute_uv=False)
    print(name, s[:45])
C=np.hstack([W['W_o.3'],W['W_proj.3']])
s=np.linalg.svd(C,compute_uv=False); print('concat L3 out', s[:45]); print('ratios s[k]/s[k+1]', (s[:44]/s[1:45]).round(2))
```