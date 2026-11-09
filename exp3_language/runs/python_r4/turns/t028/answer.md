```python
import numpy as np
hd=64
def OVl(l,j):
    lw_=W[f'ln1_w.{l}']
    return W[f'W_o.{l}'][:,j*hd:(j+1)*hd]@(W[f'W_v.{l}'][j*hd:(j+1)*hd]*lw_)
def QKl(l,j):
    lw_=W[f'ln1_w.{l}']; s=slice(j*hd,(j+1)*hd)
    return (W[f'W_q.{l}'][s]*lw_).T@(W[f'W_k.{l}'][s]*lw_)
k=24
_,_,Vt1=np.linalg.svd(OVl(0,1)); _,_,Vt5=np.linalg.svd(OVl(0,5))
Bt,_,_=np.linalg.svd(np.hstack([Vt1[:k].T,Vt5[:k].T]),full_matrices=False); B=Bt[:,:k]
mats={}
for (l,j) in [(2,1),(3,1),(3,5)]:
    A=B.T@QKl(l,j)@OVl(0,5)@B
    mats[(l,j)]=A
    sym=(A+A.T)/2
    print(l,j,'symfrac %.2f'%(np.linalg.norm(sym)**2/np.linalg.norm(A)**2), 'sym eig',np.round(np.linalg.eigvalsh(sym)[::-1][:26]/np.abs(np.linalg.eigvalsh(sym)).max(),2))
# also consider prev-token map composition: OV5 restricted -> S5; check how OV5 acts on B (singular values)
print('OV5 on B sv', np.round(np.linalg.svd(OVl(0,5)@B,compute_uv=False),2))
print('OV1 on B sv', np.round(np.linalg.svd(OVl(0,1)@B,compute_uv=False),2))
```