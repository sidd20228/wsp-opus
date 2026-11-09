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
k=24
_,_,Vt1=np.linalg.svd(OVl(0,1)); _,_,Vt5=np.linalg.svd(OVl(0,5))
Bt,_,_=np.linalg.svd(np.hstack([Vt1[:k].T,Vt5[:k].T]),full_matrices=False); B=Bt[:,:k]
QKs=np.hstack([QKl(0,1),QKl(0,5),QKl(0,1).T,QKl(0,5).T])
Pr=QKs - B@(B.T@QKs)
U,S,_=np.linalg.svd(Pr); print('pos QK spectrum',np.round(S[:30],2))
Pin=U[:,:12]
def ratio(A,S): return (np.linalg.norm(A@S)**2/np.linalg.norm(A)**2)/(S.shape[1]/384)
for l in range(4):
    lw=W[f'ln1_w.{l}']
    print(l,'V-read of Pin per head:',' '.join('%.1f'%ratio(W[f'W_v.{l}'][j*hd:(j+1)*hd]*lw,Pin) for j in range(6)),
          '| V-read of B:',' '.join('%.1f'%ratio(W[f'W_v.{l}'][j*hd:(j+1)*hd]*lw,B) for j in range(6)))
print('MLP0 reads Pin %.1f'%ratio(W['W_fc.0']*W['ln2_w.0'],Pin))
```