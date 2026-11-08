```python
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
np.set_printoptions(precision=2, suppress=True, linewidth=250)
w2=W['ln2_w.0']
rows=np.concatenate([W['W_v.0'][64*j:64*j+64] for j in [1,2,3,5]],0)
T=np.linalg.svd(rows,full_matrices=False)[2][:27].T
Wf=W['W_fc.0']*w2
F=Wf@T
nr=np.linalg.norm(F,axis=1); full=np.linalg.norm(Wf,axis=1)
idx=np.where(nr>0.6)[0]
Fn=F[idx]/nr[idx,None]
Z=linkage(Fn,'average',metric='cosine')
for thr in [0.05,0.1,0.2,0.3]:
    lab=fcluster(Z,thr,'distance')
    sizes=np.bincount(lab)[1:]
    print(thr,'nclusters',len(sizes),'sizes>=3:',np.sort(sizes[sizes>=3])[::-1])
# how much of those neurons' norm is in T
print('T frac of selected neurons', np.quantile(nr[idx]/full[idx],[.1,.5,.9]).round(2))
```