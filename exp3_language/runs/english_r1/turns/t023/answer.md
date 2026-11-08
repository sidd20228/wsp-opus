<invoke name="python">
```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
H=6; hd=64
def hw(l,j):
    w=W[f'ln1_w.{l}']; s=slice(j*hd,(j+1)*hd)
    return W[f'W_q.{l}'][s]*w, W[f'W_k.{l}'][s]*w, W[f'W_v.{l}'][s]*w, W[f'W_o.{l}'][:,s]
q,k,v,o=hw(0,2); OV2=o@v
fc=W['W_fc.0']*W['ln2_w.0']
alive=np.linalg.norm(fc,axis=1)>1.0
def prof(M,n=32):
    s=np.linalg.svd(M,compute_uv=False); return (s[:n]/s[0]).round(2)
print('fc0(alive)@OV2', prof(fc[alive]@OV2))
for j in range(6):
    q,k,v,o=hw(1,j); print('L1H%d V@OV2'%j, prof(v@OV2))
q,k,v,o=hw(3,1); print('L3H1 K@OV2', prof(k@OV2))
# ratio of consecutive
s=np.linalg.svd(fc[alive]@OV2,compute_uv=False); print('consecutive ratios', (s[1:32]/s[:31]).round(2))
```

````