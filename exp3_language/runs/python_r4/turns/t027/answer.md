```python
import numpy as np
hd=64
def OVl(l,j):
    lw_=W[f'ln1_w.{l}']
    return W[f'W_o.{l}'][:,j*hd:(j+1)*hd]@(W[f'W_v.{l}'][j*hd:(j+1)*hd]*lw_)
def QKl(l,j):
    lw_=W[f'ln1_w.{l}']; s=slice(j*hd,(j+1)*hd)
    return (W[f'W_q.{l}'][s]*lw_).T@(W[f'W_k.{l}'][s]*lw_)
def top_out(M,k): U,S,Vt=np.linalg.svd(M); return U[:,:k]
def top_in(M,k): U,S,Vt=np.linalg.svd(M); return Vt[:k].T
k=24
_,_,Vt1=np.linalg.svd(OVl(0,1)); _,_,Vt5=np.linalg.svd(OVl(0,5))
Bt,_,_=np.linalg.svd(np.hstack([Vt1[:k].T,Vt5[:k].T]),full_matrices=False); B=Bt[:,:k]
# positional input subspace: from QK of H1,H5 excluding B
QKs=np.hstack([QKl(0,1),QKl(0,5),QKl(0,1).T,QKl(0,5).T])
Pr=QKs - B@(B.T@QKs)
Pin=top_out(Pr,16)
S1=top_out(OVl(0,1),20); S5=top_out(OVl(0,5),20); P4=top_out(OVl(0,4),2)
subs={'B':B,'Pin':Pin,'S1':S1,'S5':S5,'P4':P4}
print('overlaps: B-Pin',np.linalg.norm(B.T@Pin)**2,'P4-Pin',np.linalg.norm(P4.T@Pin)**2,'S1-S5',np.linalg.norm(S1.T@S5)**2)
for l,j in [(1,0),(1,1),(1,3),(2,1),(3,1),(3,5),(2,2)]:
    QK=QKl(l,j); tot=np.linalg.norm(QK)
    print(f'L{l}H{j} block norms (row=query sub, col=key sub), /||QK||={tot:.1f}')
    names=list(subs)
    print('      '+' '.join('%6s'%n for n in names))
    for a in names:
        print('%5s '%a+' '.join('%6.3f'%(np.linalg.norm(subs[a].T@QK@subs[b])/tot) for b in names))
```