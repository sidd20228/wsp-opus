<invoke name="python">
```python
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
np.set_printoptions(precision=2, suppress=True, linewidth=220)
H=6; hd=64
def hw(l,j):
    w=W[f'ln1_w.{l}']; s=slice(j*hd,(j+1)*hd)
    return W[f'W_q.{l}'][s]*w, W[f'W_k.{l}'][s]*w, W[f'W_v.{l}'][s]*w, W[f'W_o.{l}'][:,s]
q,k,v,o=hw(0,2); OV2=o@v
fc=W['W_fc.0']*W['ln2_w.0']
alive=np.linalg.norm(fc,axis=1)>1.0
F=fc[alive]
G=F@OV2
U,S,Vt=np.linalg.svd(G,full_matrices=False)
B=G@Vt[:25].T   # prev-token coords
nb=np.linalg.norm(B,axis=1)
print('norm quantiles of prev-token part', np.quantile(nb,[.1,.5,.9,.99]).round(2))
sel=nb>np.quantile(nb,0.3)
Bn=B[sel]/nb[sel,None]
C=Bn@Bn.T
np.fill_diagonal(C,0)
print('max cos per neuron quantiles', np.quantile(C.max(1),[.1,.25,.5,.75,.9]).round(2))
Z=linkage(Bn,'average',metric='cosine')
for t in [0.1,0.2,0.3,0.4]:
    cl=fcluster(Z,t,criterion='distance')
    sizes=np.bincount(cl)[1:]
    print(t,'nclusters',len(sizes),' with>=3:',int((sizes>=3).sum()),' sizes sorted',np.sort(sizes)[::-1][:35])
```

````