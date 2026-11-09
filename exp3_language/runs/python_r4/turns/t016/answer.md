```python
import numpy as np
hd=64
lw0=W['ln1_w.0']
def OV0(j): return W['W_o.0'][:,j*hd:(j+1)*hd]@(W['W_v.0'][j*hd:(j+1)*hd]*lw0)
k=24
_,_,Vt1=np.linalg.svd(OV0(1)); _,_,Vt5=np.linalg.svd(OV0(5))
Bt,_,_=np.linalg.svd(np.hstack([Vt1[:k].T,Vt5[:k].T]),full_matrices=False); B=Bt[:,:k]
for j in range(6):
    O=OV0(j)
    U,S,Vt=np.linalg.svd(O)
    outB=np.linalg.norm(B.T@O)**2/np.linalg.norm(O)**2
    inB=np.linalg.norm(O@B)**2/np.linalg.norm(O)**2
    M=B.T@O@B
    ev=np.linalg.eigvals(M)
    print(j,'output in B %.2f, input in B %.2f, BOB eigratio %.2f norm %.2f / %.2f'%(outB,inB,ev.real.sum()/np.abs(ev).sum(),np.linalg.norm(M),np.linalg.norm(O)))
# overlap of output subspaces of H1 and H5
U1,_,_=np.linalg.svd(OV0(1)); U5,_,_=np.linalg.svd(OV0(5))
print('out overlap H1/H5', np.round(np.linalg.svd(U1[:,:24].T@U5[:,:24],compute_uv=False),2))
```