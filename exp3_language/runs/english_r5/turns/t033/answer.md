```python
import numpy as np
def rows(m,j,l,fold=True): 
    s=slice(j*64,(j+1)*64); return W[f'W_{m}.{l}'][s]*(W[f'ln1_w.{l}'] if fold else 1)
s=slice(64,128)
OV=W['W_o.3'][:,s]@rows('v',1,3)
print('L3H1 OV sv',np.linalg.svd(OV,compute_uv=False).round(2)[:40])
u,sv,vt=np.linalg.svd(W['W_proj.3'],full_matrices=False)
U=u[:,:28]
print('L3H1 OV in U28 sv',np.linalg.svd(U.T@OV,compute_uv=False).round(2))
print('L3H1 Wv sv',np.linalg.svd(W['W_v.3'][s],compute_uv=False).round(2)[:40])
print('L3H1 Wo sv',np.linalg.svd(W['W_o.3'][:,s],compute_uv=False).round(2)[:40])
```