```python
import numpy as np
np.set_printoptions(linewidth=220)
hd=64; w0=W['ln1_w.0']
for j in [0,3,5,1]:
    s=slice(j*hd,(j+1)*hd)
    for nm in ['W_v','W_q','W_k']:
        M=W[f'{nm}.0'][s]*w0
        print(j,nm,np.round(np.linalg.svd(M,compute_uv=False),2))
    print(j,'OV',np.round(np.linalg.svd(W['W_o.0'][:,s]@(W['W_v.0'][s]*w0),compute_uv=False)[:40],2))
print('fc (ln2 scaled)',np.round(np.linalg.svd(W['W_fc.0']*W['ln2_w.0'],compute_uv=False)[:80],2))
```