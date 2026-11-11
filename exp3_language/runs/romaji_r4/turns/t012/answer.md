```python
import numpy as np
fro=np.linalg.norm
H=6;hd=64
def head(l,j):
    g=W[f'ln1_w.{l}'];s=slice(j*hd,(j+1)*hd)
    return W[f'W_q.{l}'][s]*g, W[f'W_k.{l}'][s]*g, W[f'W_v.{l}'][s]*g, W[f'W_o.{l}'][:,s]
Up,sp,_=np.linalg.svd(W['W_proj.3'],full_matrices=False)
Q,K,V,O=head(3,3);OV=O@V
uo,so,vo=np.linalg.svd(OV)
print('L3H3 full OV sv:',np.round(so[:40],2))
print('L3H3 write energy per W_proj.3 dir:',np.round([fro(Up[:,i]@OV) for i in range(36)],2))
for k in [24,25,26,27,30]:
    print('L3H3 top-%d out dirs: frac in W_proj.3 top-28 %.2f, top-40 %.2f'%(k,fro(Up[:,:28].T@uo[:,:k])**2/k,fro(Up[:,:40].T@uo[:,:k])**2/k))
Q4,K4,V4,O4=head(3,4);OV4=O4@V4
print('L3H4 write energy per W_proj.3 dir:',np.round([fro(Up[:,i]@OV4) for i in range(36)],2))
```