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
U,S,_=np.linalg.svd(Pr); Pin=U[:,:24]
R=np.eye(384)-B@B.T-Pin@Pin.T
lw=W['ln1_w.0']; lb=W['ln1_b.0']
for j in range(6):
    QK=QKl(j and j or 0, j) if False else QKl(0,j)
    tot=np.linalg.norm(QK)
    bb=np.linalg.norm(B.T@QK@B)/tot; bp=np.linalg.norm(B.T@QK@Pin)/tot; pb=np.linalg.norm(Pin.T@QK@B)/tot; pp=np.linalg.norm(Pin.T@QK@Pin)/tot
    rest=np.linalg.norm(R@QK@R)/tot
    s=slice(j*hd,(j+1)*hd)
    qb=W['W_q.0'][s]@lb; kp=(W['W_k.0'][s]*lw).T@qb
    print('L0H%d |QK|=%.2f  qB-kB %.2f qB-kP %.2f qP-kB %.2f qP-kP %.2f rest %.2f | key-pref: B %.2f P %.2f (|kp| %.2f)'%(j,tot,bb,bp,pb,pp,rest,np.linalg.norm(B.T@kp),np.linalg.norm(Pin.T@kp),np.linalg.norm(kp)))
```