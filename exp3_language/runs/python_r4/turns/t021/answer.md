```python
import numpy as np
hd=64
lw=W['ln1_w.0']
def OVl(l,j):
    lw_=W[f'ln1_w.{l}']
    return W[f'W_o.{l}'][:,j*hd:(j+1)*hd]@(W[f'W_v.{l}'][j*hd:(j+1)*hd]*lw_)
k=24
_,_,Vt1=np.linalg.svd(OVl(0,1)); _,_,Vt5=np.linalg.svd(OVl(0,5))
Bt,_,_=np.linalg.svd(np.hstack([Vt1[:k].T,Vt5[:k].T]),full_matrices=False); B=Bt[:,:k]
def QKh(j):
    s=slice(j*hd,(j+1)*hd)
    return (W['W_q.0'][s]*lw).T@(W['W_k.0'][s]*lw)
for j in [1,5,0,2,3]:
    M=QKh(j)
    ev,vec=np.linalg.eig(M)
    idx=np.argsort(-np.abs(ev))
    out=[]
    cnt=0
    for i in idx:
        e=ev[i]
        if e.imag<-1e-9: continue
        v=vec[:,i]
        Bm=np.stack([v.real,v.imag],1) if abs(e.imag)>1e-9 else v.real[:,None]
        Q,_=np.linalg.qr(Bm)
        out.append('%.2f@%d(B%.2f)'%(abs(e),np.degrees(np.angle(e)),np.linalg.norm(B.T@Q)**2/Q.shape[1]))
        cnt+=1
        if cnt>=10: break
    print(j,' '.join(out))
```