```python
import numpy as np
hd=64
def OVl(l,j):
    lw_=W[f'ln1_w.{l}']
    return W[f'W_o.{l}'][:,j*hd:(j+1)*hd]@(W[f'W_v.{l}'][j*hd:(j+1)*hd]*lw_)
def QKl(l,j):
    lw_=W[f'ln1_w.{l}']; s=slice(j*hd,(j+1)*hd)
    return (W[f'W_q.{l}'][s]*lw_).T@(W[f'W_k.{l}'][s]*lw_)
k=36
_,_,Vt1=np.linalg.svd(OVl(0,1)); _,_,Vt5=np.linalg.svd(OVl(0,5))
Bt,Sb,_=np.linalg.svd(np.hstack([Vt1[:k].T*np.linalg.svd(OVl(0,1),compute_uv=False)[:k],Vt5[:k].T*np.linalg.svd(OVl(0,5),compute_uv=False)[:k]]),full_matrices=False)
print('combined sv',np.round(Sb[:40],2))
B=Bt[:,:k]
for (l,j,typ) in [(1,1,'same'),(1,3,'same'),(2,1,'ind'),(3,5,'ind'),(3,1,'ind'),(2,2,'same')]:
    M=B.T@QKl(l,j)@(OVl(0,5) if typ=='ind' else np.eye(384))@B
    s=np.linalg.svd(M,compute_uv=False)
    print(l,j,typ,np.round(s[:36]/s[0],2))
```