```python
import numpy as np
hd=64
lw=W['ln1_w.0']
def Vh(j): return W['W_v.0'][j*hd:(j+1)*hd]*lw
def OVh(j): return W['W_o.0'][:,j*hd:(j+1)*hd]@Vh(j)
for js in [[1],[5],[1,5],[1,2,3,5]]:
    M=np.vstack([OVh(j) for j in js])
    s=np.linalg.svd(M,compute_uv=False)
    print(js,np.round(s[:40],2))
```