```python
import numpy as np
np.set_printoptions(linewidth=220)
hd=64
sl=lambda j: slice(j*hd,(j+1)*hd)
def pc(U1,U2): return np.linalg.svd(U1@U2.T,compute_uv=False)
up3,_,_=np.linalg.svd(W['W_proj.3'],full_matrices=False)
up2,_,_=np.linalg.svd(W['W_proj.2'],full_matrices=False)
OV33=W['W_o.3'][:,sl(3)]@(W['W_v.3'][sl(3)]*W['ln1_w.3']); u33,_,_=np.linalg.svd(OV33)
OV34=W['W_o.3'][:,sl(4)]@(W['W_v.3'][sl(4)]*W['ln1_w.3']); u34,_,_=np.linalg.svd(OV34)
for r1,r2 in [(30,30),(30,50),(30,80),(34,100)]:
    print('Wp3',r1,'Wp2',r2,np.round(pc(up3[:,:r1].T,up2[:,:r2].T),2))
for r1,r2 in [(30,40),(30,60)]:
    print('L3H3',r1,'Wp2',r2,np.round(pc(u33[:,:r1].T,up2[:,:r2].T),2))
# random baseline for principal cosines of 30-dim vs 50-dim random subspaces in 384
rng=np.random.default_rng(0)
Q1=np.linalg.qr(rng.normal(size=(384,30)))[0].T;Q2=np.linalg.qr(rng.normal(size=(384,50)))[0].T
print('random 30 vs 50',np.round(pc(Q1,Q2)[:10],2))
Q2=np.linalg.qr(rng.normal(size=(384,80)))[0].T
print('random 30 vs 80',np.round(pc(Q1,Q2)[:10],2))
```