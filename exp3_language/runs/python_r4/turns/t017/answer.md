```python
import numpy as np
hd=64
lw0=W['ln1_w.0']
def Vh(j): return W['W_v.0'][j*hd:(j+1)*hd]*lw0
def OV0(j): return W['W_o.0'][:,j*hd:(j+1)*hd]@Vh(j)
for k in [20,22,24,26,28,30,34]:
    _,_,Vt1=np.linalg.svd(OV0(1)); _,_,Vt5=np.linalg.svd(OV0(5))
    c=np.linalg.svd(Vt1[:k]@Vt5[:k].T,compute_uv=False)
    print(k,np.round(c,2))
```