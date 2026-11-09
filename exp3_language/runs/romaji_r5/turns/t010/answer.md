```python
import numpy as np
np.set_printoptions(linewidth=220)
hd=64; w0=W['ln1_w.0']
def topRows(M,r):
    u,s,vt=np.linalg.svd(M,full_matrices=False); return vt[:r]
def pc(U1,U2): return np.linalg.svd(U1@U2.T,compute_uv=False)
v3=W['W_v.0'][192:256]*w0; v0=W['W_v.0'][0:64]*w0
q3=W['W_q.0'][192:256]*w0; k3=W['W_k.0'][192:256]*w0
fc=W['W_fc.0']*W['ln2_w.0']
for r1,r2 in [(20,40),(30,40),(30,60),(40,80),(64,100)]:
    print('v3 vs fc',r1,r2,np.round(pc(topRows(v3,r1),topRows(fc,r2)),2))
for r1,r2 in [(30,30),(64,64)]:
    print('v0 vs v3',r1,r2,np.round(pc(topRows(v0,r1),topRows(v3,r2)),2))
    print('q3 vs v3',r1,r2,np.round(pc(topRows(q3,r1),topRows(v3,r2)),2))
    print('q3 vs k3',r1,r2,np.round(pc(topRows(q3,r1),topRows(k3,r2)),2))
    print('q3 vs fc',r1,80,np.round(pc(topRows(q3,r1),topRows(fc,80)),2))
```