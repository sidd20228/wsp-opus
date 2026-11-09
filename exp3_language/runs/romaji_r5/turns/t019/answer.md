```python
import numpy as np
np.set_printoptions(linewidth=220)
hd=64
sl=lambda j: slice(j*hd,(j+1)*hd)
up,sp,_=np.linalg.svd(W['W_proj.3'],full_matrices=False)
# other writers
OV3all=np.hstack([W['W_o.3'][:,sl(h)]@(W['W_v.3'][sl(h)]*W['ln1_w.3']) for h in range(6)])
uo,so,_=np.linalg.svd(OV3all,full_matrices=False)
u2,s2,_=np.linalg.svd(W['W_proj.2'],full_matrices=False)
u1,s1,_=np.linalg.svd(W['W_proj.1'],full_matrices=False)
print('OV3all sv',np.round(so[:40],1))
for nm,UU in [('OV3all top40',uo[:,:40]),('Wproj2 top40',u2[:,:40]),('Wproj1 top40',u1[:,:40])]:
    e=np.linalg.norm(UU.T@up[:,:40],axis=0)**2
    print(nm,'energy of Wproj3 dirs:',np.round(e,2))
# random baseline: 40/384
print('baseline',40/384)
# reverse: OV3all top dirs energy in Wproj3 top40
e=np.linalg.norm(up[:,:40].T@uo[:,:40],axis=0)**2
print('OV3all dirs energy in Wproj3 top40:',np.round(e,2))
```