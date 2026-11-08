```python
import numpy as np
H=6;hd=64
d0=W['ln1_w.0']
def rs(M,k):  # top-k right singular subspace
    U,S,Vt=np.linalg.svd(M); return Vt[:k].T, S
for h in range(H):
    s=slice(h*hd,(h+1)*hd)
    for nm in ['W_q','W_k','W_v']:
        M=W[nm+'.0'][s]*d0
        sv=np.linalg.svd(M,compute_uv=False)
        print(h,nm,np.round(sv[:30],2))
```