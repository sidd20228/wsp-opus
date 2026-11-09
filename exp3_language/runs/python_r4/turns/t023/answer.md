```python
import numpy as np
hd=64
def OVl(l,j):
    lw_=W[f'ln1_w.{l}']
    return W[f'W_o.{l}'][:,j*hd:(j+1)*hd]@(W[f'W_v.{l}'][j*hd:(j+1)*hd]*lw_)
k=24
_,_,Vt1=np.linalg.svd(OVl(0,1)); _,_,Vt5=np.linalg.svd(OVl(0,5))
Bt,_,_=np.linalg.svd(np.hstack([Vt1[:k].T,Vt5[:k].T]),full_matrices=False); B=Bt[:,:k]
KP=[];names=[]
for l in range(4):
    lw=W[f'ln1_w.{l}']; lb=W[f'ln1_b.{l}']
    for j in range(6):
        s=slice(j*hd,(j+1)*hd)
        qb=W[f'W_q.{l}'][s]@lb
        kp=(W[f'W_k.{l}'][s]*lw).T@qb
        kb=B.T@kp
        KP.append(kb); names.append(f'L{l}H{j}')
        print(f'L{l}H{j}','|kp| %.2f  in B %.2f (%.2f of energy)'%(np.linalg.norm(kp),np.linalg.norm(kb),np.linalg.norm(kb)**2/np.linalg.norm(kp)**2))
KP=np.array(KP)
Kn=KP/np.linalg.norm(KP,axis=1,keepdims=True)
C=Kn@Kn.T
np.set_printoptions(linewidth=200)
print(np.round(C,2))
```