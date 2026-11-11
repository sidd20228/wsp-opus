```python
import numpy as np
def rows(m,j,l): 
    s=slice(j*64,(j+1)*64); return W[f'W_{m}.{l}'][s]*W[f'ln1_w.{l}']
D=W['ln1_w.0']
M=np.vstack([rows('q',2,0),rows('k',2,0)])
_,sv,vt=np.linalg.svd(M,full_matrices=False); P=vt[:20].T
pe=(P**2).sum(1)
idx=np.argsort(-D)
print('P energy in top70 D dims',pe[idx[:70]].sum().round(2),'of',20)
print('P energy in next 100',pe[idx[70:170]].sum().round(2),'rest',pe[idx[170:]].sum().round(2))
# how does that compare to uniform
print('uniform share', 20*70/384, 20*100/384, 20*214/384)
# raw (unscaled) Wq Wk column norms for head2 in top-70 vs rest
Wq=W['W_q.0'][128:192];Wk=W['W_k.0'][128:192]
cq=np.linalg.norm(Wq,axis=0);ck=np.linalg.norm(Wk,axis=0)
print('raw col norm q2 top70 mean',cq[idx[:70]].mean(),'rest',cq[idx[70:]].mean())
print('raw col norm k2 top70 mean',ck[idx[:70]].mean(),'rest',ck[idx[70:]].mean())
print('corr D vs 1/colnorm', np.corrcoef(D,cq)[0,1])
import numpy as np
print(np.round(np.c_[D[idx][::10],cq[idx][::10],ck[idx][::10]],2).T)
```