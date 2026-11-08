```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
w1=W['ln1_w.0']
def rs(M,k):
    U,S,Vt=np.linalg.svd(M,full_matrices=False)
    return Vt[:k].T, S
mats={}
for j in range(6):
    s=slice(64*j,64*j+64)
    for n in ['W_q','W_k','W_v']:
        mats[f'{n[-1]}{j}']=W[f'{n}.0'][s]*w1
for k,v in mats.items():
    S=np.linalg.svd(v,compute_uv=False)
    print(k, S[:12].round(2), 'sum sq top5/all', round(float((S[:5]**2).sum()/(S**2).sum()),2))
```