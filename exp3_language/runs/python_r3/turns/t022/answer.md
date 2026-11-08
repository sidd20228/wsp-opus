```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
w0=W['ln1_w.0']
rows=np.concatenate([W['W_v.0'][64*j:64*j+64] for j in [1,2,3,5]],0)
T=np.linalg.svd(rows,full_matrices=False)[2][:27].T
for L in range(4):
    wl=W[f'ln1_w.{L}']
    for h in range(6):
        s=slice(64*h,64*h+64)
        K=W[f'W_k.{L}'][s]*wl; Q=W[f'W_q.{L}'][s]*wl
        M=Q.T@K
        MT=T.T@M@T
        sk=np.linalg.svd(K@T,compute_uv=False); sq=np.linalg.svd(Q@T,compute_uv=False); sm=np.linalg.svd(MT,compute_uv=False)
        print(f'L{L}H{h} K@T {sk[:5]} Q@T {sq[:5]} M_TT {sm[:5]}')
```