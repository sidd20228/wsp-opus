```python
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=200)
for k in sorted(W.keys()):
    a = W[k]
    print(k, a.shape, 'mean %.4f std %.4f absmax %.4f' % (a.mean(), a.std(), np.abs(a).max()))
print()
for l in range(4):
    print('layer', l)
    for j in range(6):
        s = slice(64*j, 64*j+64)
        q = W[f'W_q.{l}'][s]; k = W[f'W_k.{l}'][s]; v = W[f'W_v.{l}'][s]; o = W[f'W_o.{l}'][:, s]
        print(' head', j, 'q %.2f k %.2f v %.2f o %.2f' % tuple(np.linalg.norm(m) for m in (q,k,v,o)),
              ' qk sv:', np.linalg.svd(q.T@k, compute_uv=False)[:6], ' ov sv:', np.linalg.svd(o@v, compute_uv=False)[:6])
```