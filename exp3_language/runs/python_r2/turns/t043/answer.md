```python
import numpy as np
hd=64
sl=lambda h: slice(h*hd,(h+1)*hd)
def rsvd(M):
    U,S,Vt=np.linalg.svd(M,full_matrices=False); return Vt,S
V1,s1=rsvd(W['W_v.0'][sl(1)])
for h in [5,2,3,0]:
    r=[np.linalg.norm(W['W_v.0'][sl(h)]@V1[i]) for i in range(30)]
    print('H%d response to H1 V dirs:'%h,np.round(r,2))
# MLP0 response (alive) to H1 V dirs (in h-coords of ln1 -> need x-hat functionals; approximate using D0 and ln2_w)
D0=W['ln1_w.0'];D2=W['ln2_w.0']
Fc=W['W_fc.0']; alive=np.linalg.norm(Fc*D2,axis=1)>0.6
r=[]
for i in range(30):
    # direction in x-hat space corresponding to h-direction v: xhat-direction u with D0*u ~ v  -> u = v/D0 (ill-cond). Instead use functional match: skip
    pass
print('H1 sv',np.round(s1[:30],2))
```