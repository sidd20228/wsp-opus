```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
def hd(l,j):
    w = W[f'ln1_w.{l}']; s = slice(64*j, 64*j+64)
    return W[f'W_q.{l}'][s]*w, W[f'W_k.{l}'][s]*w, W[f'W_v.{l}'][s]*w, W[f'W_o.{l}'][:, s]
for (l,j) in [(3,3),(3,4),(3,1),(0,3),(0,0),(2,4)]:
    q,k,v,o = hd(l,j)
    print(f'L{l}H{j} OV sv', np.linalg.svd(o@v, compute_uv=False))
    print(f'L{l}H{j} V  sv', np.linalg.svd(v, compute_uv=False))
```