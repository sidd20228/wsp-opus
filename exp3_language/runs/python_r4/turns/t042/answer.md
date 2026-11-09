```python
import numpy as np
from sklearn.decomposition import FastICA
hd=64
def OVl(l,j):
    lw_=W[f'ln1_w.{l}']
    return W[f'W_o.{l}'][:,j*hd:(j+1)*hd]@(W[f'W_v.{l}'][j*hd:(j+1)*hd]*lw_)
U,S,_=np.linalg.svd(W['W_proj.3'],full_matrices=False)
# energy of copy-head outputs in each W_proj.3 singular direction
for (l,j) in [(3,5),(3,1),(3,0),(3,2),(3,3),(3,4)]:
    O=OVl(l,j)
    e=np.linalg.norm(U[:,:40].T@O,axis=1)**2/np.linalg.norm(O)**2*384
    print(f'L{l}H{j}',' '.join('%.1f'%x for x in e))
# also W_o.3 overall and W_proj.2
for nm in ['W_o.3','W_proj.2','W_o.2']:
    O=W[nm]; e=np.linalg.norm(U[:,:40].T@O,axis=1)**2/np.linalg.norm(O)**2*384
    print(nm,' '.join('%.1f'%x for x in e))
```