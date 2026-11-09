```python
import numpy as np
np.set_printoptions(linewidth=220)
hd=64; w0=W['ln1_w.0']
def topRows(M,r):
    u,s,vt=np.linalg.svd(M,full_matrices=False); return vt[:r]
def pc(U1,U2): return np.linalg.svd(U1@U2.T,compute_uv=False)
sl=lambda j: slice(j*hd,(j+1)*hd)
v3=W['W_v.0'][sl(3)]*w0; v0=W['W_v.0'][sl(0)]*w0
q3=W['W_q.0'][sl(3)]*w0; k3=W['W_k.0'][sl(3)]*w0
fc=W['W_fc.0']*W['ln2_w.0']
# positional subspace: common subspace of q3,k3
U=np.vstack([topRows(q3,30),topRows(k3,30)])
u,s,vt=np.linalg.svd(U); print('q3+k3 combined sv',np.round(s[:40],2))
for npos in [20,26,30]:
    P=vt[:npos]
    Proj=np.eye(384)-P.T@P
    v3p=v3@Proj; v0p=v0@Proj; fcp=fc@Proj
    print('npos',npos,'v3 sv after removing P',np.round(np.linalg.svd(v3p,compute_uv=False)[:32],2))
    for r1,r2 in [(25,50),(32,60),(40,80)]:
        print('   v3p vs fcp',r1,r2,np.round(pc(topRows(v3p,r1),topRows(fcp,r2)),2))
    print('   v0p vs v3p 32',np.round(pc(topRows(v0p,32),topRows(v3p,32)),2))
```