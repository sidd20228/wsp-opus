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
QKs=np.hstack([QKl(0,1),QKl(0,5),QKl(0,1).T,QKl(0,5).T])
Pr=QKs - B@(B.T@QKs)
U,S,_=np.linalg.svd(Pr); Pin=U[:,:16]
def phases(M,n=8):
    ev=np.linalg.eigvals(M); idx=np.argsort(-np.abs(ev))
    out=[];c=0
    for i in idx:
        if ev[i].imag<-1e-9: continue
        out.append('%.2f@%d'%(abs(ev[i]),round(np.degrees(np.angle(ev[i])))));c+=1
        if c>=n: break
    return ' '.join(out)
for l in range(4):
    for j in range(6):
        QK=QKl(l,j)
        M=Pin.T@QK@Pin
        print(f'L{l}H{j} |Pin QK Pin|=%.2f (frac %.3f)  '%(np.linalg.norm(M),np.linalg.norm(M)**2/np.linalg.norm(QK)**2*384/16), phases(M))
```