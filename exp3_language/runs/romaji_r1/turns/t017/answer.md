```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=200)
def hd(l,j):
    w = W[f'ln1_w.{l}']; s = slice(64*j, 64*j+64)
    return W[f'W_q.{l}'][s]*w, W[f'W_k.{l}'][s]*w, W[f'W_v.{l}'][s]*w, W[f'W_o.{l}'][:, s]
def rows(m, r): return np.linalg.svd(m)[2][:r]
q3,k3,v3,o3 = hd(0,3)
S = rows(np.vstack([q3/np.linalg.norm(q3), k3/np.linalg.norm(k3)]), 20)
T = rows(v3, 24)
for l in range(1,4):
    for j in range(6):
        q,k,v,o = hd(l,j)
        G = T@(q.T@k)@T.T
        sv = np.linalg.svd(G, compute_uv=False)
        es = np.linalg.eigvalsh((G+G.T)/2)
        print(f'L{l}H{j}: TT sv', sv[:6], ' sym eig low', es[:3], 'high', es[-3:])
```