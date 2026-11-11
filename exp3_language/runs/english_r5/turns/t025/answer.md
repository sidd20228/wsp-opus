```python
import numpy as np
def rows(m,j,l): 
    s=slice(j*64,(j+1)*64); return W[f'W_{m}.{l}'][s]*W[f'ln1_w.{l}']
def OV(l,j):
    s=slice(j*64,(j+1)*64); return W[f'W_o.{l}'][:,s]@rows('v',j,l)
def QK(l,j): return rows('q',j,l).T@rows('k',j,l)
A=QK(3,1); O=OV(0,2)
M=A@O
ev,evec=np.linalg.eig(M)
o=np.argsort(-np.abs(ev))[:20]
E=evec[:,o]
# real basis of top eig space
B=np.linalg.qr(np.hstack([E.real,E.imag]))[0][:,:20]
# compare: V read space of L3H1, Q-read space of L3H1, V read of L0H2
def topr(M,k):
    _,_,vt=np.linalg.svd(M,full_matrices=False); return vt[:k].T
for name,X in [('V31',topr(rows('v',1,3),20)),('Q31',topr(rows('q',1,3),20)),('K31',topr(rows('k',1,3),20)),('V02',topr(rows('v',2,0),20)),('P',topr(np.vstack([rows('q',2,0),rows('k',2,0)]),20)),('U23',np.linalg.svd(W['W_proj.3'],full_matrices=False)[0][:,:23])]:
    s=np.linalg.svd(B.T@X,compute_uv=False)
    print(name,s.round(2)[:12],'sumsq',(s**2).sum().round(2),'baseline',round(20*X.shape[1]/384,2))
```