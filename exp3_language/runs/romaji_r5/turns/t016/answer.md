```python
import numpy as np
np.set_printoptions(linewidth=220)
hd=64
sl=lambda j: slice(j*hd,(j+1)*hd)
def pc(U1,U2): return np.linalg.svd(U1@U2.T,compute_uv=False)
up,sp,_=np.linalg.svd(W['W_proj.3'],full_matrices=False)
Pr=up[:,:40].T
for h in range(6):
    OVh=W['W_o.3'][:,sl(h)]@(W['W_v.3'][sl(h)]*W['ln1_w.3'])
    uu,ss,_=np.linalg.svd(OVh)
    print(f'L3H{h} OV sv',np.round(ss[:34],1))
    c=np.linalg.norm(Pr@uu[:,:30],axis=0)   # how much each OV output dir lies in top-40 Wproj3 space
    print('     frac of OV out-dirs in top40 Wproj3:',np.round(c,2))
# energy of each L3 OV in Wproj3 top-r subspaces
for r in [20,24,28,32,40]:
    Q=up[:,:r]
    row=[]
    for h in range(6):
        OVh=W['W_o.3'][:,sl(h)]@(W['W_v.3'][sl(h)]*W['ln1_w.3'])
        row.append(np.linalg.norm(Q.T@OVh)**2/np.linalg.norm(OVh)**2)
    print(r,np.round(row,2))
```