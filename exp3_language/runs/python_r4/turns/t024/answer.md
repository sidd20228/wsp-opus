```python
import numpy as np
hd=64
def OVl(l,j):
    lw_=W[f'ln1_w.{l}']
    return W[f'W_o.{l}'][:,j*hd:(j+1)*hd]@(W[f'W_v.{l}'][j*hd:(j+1)*hd]*lw_)
k=24
_,_,Vt1=np.linalg.svd(OVl(0,1)); _,_,Vt5=np.linalg.svd(OVl(0,5))
Bt,_,_=np.linalg.svd(np.hstack([Vt1[:k].T,Vt5[:k].T]),full_matrices=False); B=Bt[:,:k]
U,S,Vt=np.linalg.svd(W['W_proj.3'],full_matrices=False)
Ul=U[:,:27]
print('B vs logit subspace cosines', np.round(np.linalg.svd(B.T@Ul,compute_uv=False),2))
# OV of later heads restricted: from B (input) to Ul (output)
for l in [1,2,3]:
    for j in range(6):
        O=OVl(l,j)
        M=Ul.T@O@B
        Mb=B.T@O@B
        ev=np.linalg.eigvals(Mb)
        print(l,j,'||U^T OV B||=%.2f  ||B^T OV B||=%.2f eigratio(BOB)=%.2f  ||OV||=%.2f'%(np.linalg.norm(M),np.linalg.norm(Mb),ev.real.sum()/np.abs(ev).sum(),np.linalg.norm(O)))
```