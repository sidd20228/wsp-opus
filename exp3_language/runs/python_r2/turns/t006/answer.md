```python
import numpy as np
H=6;hd=64
U,S,Vt=np.linalg.svd(W['W_o.0'])
print(S[:5])
for i in range(3):
    u=U[:,i]; v=Vt[i]
    print('dir',i,'head contributions (col norms of v per head):',[round(float(np.linalg.norm(v[h*hd:(h+1)*hd])),3) for h in range(H)])
    # top residual dims
    idx=np.argsort(-np.abs(u))[:8]; print('  top resid dims',idx, np.round(u[idx],3))
# per-head OV singular values
for l in range(4):
    for h in range(H):
        s=slice(h*hd,(h+1)*hd)
        M=W[f'W_o.{l}'][:,s]@(W[f'W_v.{l}'][s]*W[f'ln1_w.{l}'])
        sv=np.linalg.svd(M,compute_uv=False)
        print(l,h,np.round(sv[:8],2))
```