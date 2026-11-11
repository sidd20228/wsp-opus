```python
import numpy as np
D=W['ln1_w.0']
def rows(m,j): 
    s=slice(j*64,(j+1)*64); return W[f'W_{m}.0'][s]*D
def topspace(M,k):
    u,s,vt=np.linalg.svd(M,full_matrices=False); return vt[:k],s
Vb=np.vstack([rows('v',j) for j in [0,1,4,5]])
for name,M in [('Vbroad',Vb),('q2',rows('q',2)),('k2',rows('k',2)),('q3',rows('q',3)),('k3',rows('k',3)),('v2',rows('v',2)),('v3',rows('v',3)),('fc',W['W_fc.0']*W['ln2_w.0'])]:
    _,s=topspace(M,1)
    print(name, np.round(s[:40],2))
```