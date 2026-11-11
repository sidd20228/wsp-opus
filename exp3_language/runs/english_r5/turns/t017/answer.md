```python
import numpy as np
u,s,vt=np.linalg.svd(W['W_proj.3'],full_matrices=False)
U23=u[:,:23];Um=u[:,23:28];U30=u[:,28:30];Ur=u[:,30:]
def rows(m,j,l): 
    s=slice(j*64,(j+1)*64); return W[f'W_{m}.{l}'][s]*W[f'ln1_w.{l}']
def OV(l,j):
    s=slice(j*64,(j+1)*64); return W[f'W_o.{l}'][:,s]@rows('v',j,l)
def fr(M):
    n=np.linalg.norm(M)**2
    return [round(np.linalg.norm(X.T@M)**2/n,3) for X in (U23,Um,U30)]
print('dims fractions random baseline', 23/384, 5/384, 2/384)
for l in range(4):
    print(f'proj{l}',fr(W[f'W_proj.{l}']), 'Wo',fr(W[f'W_o.{l}']))
    for j in range(6):
        print(f'  L{l}H{j}',fr(OV(l,j)))
```