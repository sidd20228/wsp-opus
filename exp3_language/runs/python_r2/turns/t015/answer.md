```python
import numpy as np
D0=W['ln1_w.0'];hd=64
for h in range(6):
    s=slice(h*hd,(h+1)*hd)
    M=W['W_o.0'][:,s]@(W['W_v.0'][s]*D0)
    sv=np.linalg.svd(M,compute_uv=False)
    print(h,np.round(sv[:40],2))
```