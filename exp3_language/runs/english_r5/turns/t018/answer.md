```python
import numpy as np
def rows(m,j,l): 
    s=slice(j*64,(j+1)*64); return W[f'W_{m}.{l}'][s]*W[f'ln1_w.{l}']
M=np.vstack([rows('q',2,0),rows('k',2,0)])
u,s,vt=np.linalg.svd(M,full_matrices=False)
P=vt[:24].T
Vb=np.vstack([rows('v',j,0) for j in [0,1,4,5]])
_,_,vtb=np.linalg.svd(Vb,full_matrices=False)
OVb=np.hstack([W['W_o.0'][:,j*64:(j+1)*64]@rows('v',j,0) for j in [0,1,4,5]])
ub,_,_=np.linalg.svd(OVb,full_matrices=False)
X=np.hstack([P,vtb[:3].T,ub[:,:3]])
Q,_=np.linalg.qr(X)
Proj=np.eye(384)-Q@Q.T
F=W['W_fc.0']*W['ln2_w.0']
s1=np.linalg.svd(F,compute_uv=False)
s2=np.linalg.svd(F@Proj,compute_uv=False)
print('fc0 full',s1[:50].round(2))
print('fc0 noP ',s2[:50].round(2))
# also raw without ln2 weight
s3=np.linalg.svd(W['W_fc.0']@Proj,compute_uv=False)
print('fc0 raw noP',s3[:50].round(2))
```