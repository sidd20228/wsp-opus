```python
import numpy as np
hd=64
def OVl(l,j):
    lw_=W[f'ln1_w.{l}']
    return W[f'W_o.{l}'][:,j*hd:(j+1)*hd]@(W[f'W_v.{l}'][j*hd:(j+1)*hd]*lw_)
def QKl(l,j):
    lw_=W[f'ln1_w.{l}']; s=slice(j*hd,(j+1)*hd)
    return (W[f'W_q.{l}'][s]*lw_).T@(W[f'W_k.{l}'][s]*lw_)
def er(M):
    ev=np.linalg.eigvals(M); return ev.real.sum()/np.abs(ev).sum()
rng=np.random.default_rng(0)
for k in [16,20,24,28]:
    _,_,Vt1=np.linalg.svd(OVl(0,1)); _,_,Vt5=np.linalg.svd(OVl(0,5))
    Bt,_,_=np.linalg.svd(np.hstack([Vt1[:k].T,Vt5[:k].T]),full_matrices=False); B=Bt[:,:k]
    res=[]
    for (l,j) in [(2,1),(3,1),(3,5),(3,0),(2,5)]:
        res.append('L%dH%d %.2f'%(l,j,er(B.T@QKl(l,j)@OVl(0,5)@B)))
    # null: random subspaces
    nulls=[]
    for _ in range(20):
        R,_=np.linalg.qr(rng.standard_normal((384,k)))
        nulls.append(er(R.T@QKl(2,1)@OVl(0,5)@R))
    print(k,res,' null L2H1 mean %.2f sd %.2f'%(np.mean(nulls),np.std(nulls)))
```