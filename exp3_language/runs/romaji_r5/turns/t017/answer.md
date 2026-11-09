```python
import numpy as np
np.set_printoptions(linewidth=220)
hd=64
sl=lambda j: slice(j*hd,(j+1)*hd)
def pc(U1,U2): return np.linalg.svd(U1@U2.T,compute_uv=False)
up,sp,_=np.linalg.svd(W['W_proj.3'],full_matrices=False)
OV33=W['W_o.3'][:,sl(3)]@(W['W_v.3'][sl(3)]*W['ln1_w.3'])
u3,s3,_=np.linalg.svd(OV33)
for r1,r2 in [(24,30),(28,34),(30,40),(34,50)]:
    print('Wproj3 top',r1,'vs L3H3 out top',r2,np.round(pc(up[:,:r1].T,u3[:,:r2].T),2))
OV34=W['W_o.3'][:,sl(4)]@(W['W_v.3'][sl(4)]*W['ln1_w.3'])
u4,s4,_=np.linalg.svd(OV34)
print('Wproj3 top30 vs L3H4 out top20',np.round(pc(up[:,:30].T,u4[:,:20].T),2))
print('L3H3 out top30 vs L3H4 out top20',np.round(pc(u3[:,:30].T,u4[:,:20].T),2))
# W_proj.2 vs W_proj.3
u2p,s2p,_=np.linalg.svd(W['W_proj.2'],full_matrices=False)
print('Wproj3 top30 vs Wproj2 top60',np.round(pc(up[:,:30].T,u2p[:,:60].T),2))
```