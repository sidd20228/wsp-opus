<invoke name="python">
```python
import numpy as np
np.set_printoptions(precision=1, suppress=True, linewidth=220)
H=6; hd=64
def hw(l,j):
    w=W[f'ln1_w.{l}']; s=slice(j*hd,(j+1)*hd)
    return W[f'W_q.{l}'][s]*w, W[f'W_k.{l}'][s]*w, W[f'W_v.{l}'][s]*w, W[f'W_o.{l}'][:,s]
for L in [1,2,3]:
    for j in range(6):
        q,k,v,o=hw(L,j); QK=q.T@k
        u,s,vt=np.linalg.svd(QK)
        ov=np.linalg.svd(o@v,compute_uv=False)
        ov_lr=np.linalg.svd(o@v)
        print(L,j,'QK sv*48',(s[:10]*48).round(0),' OV sv',ov[:8].round(1))
```

````