```python
import numpy as np
from sklearn.decomposition import FastICA
from scipy.stats import kurtosis, skew
Up,sp,Vt=np.linalg.svd(W['W_proj.3'],full_matrices=False)
for k in [24,28,32]:
    Y=(Up[:,:k].T@W['W_proj.3']).T   # neurons x k
    for seed in range(2):
        ica=FastICA(n_components=k,random_state=seed,max_iter=3000,whiten='unit-variance')
        S=ica.fit_transform(Y)
        ku=kurtosis(S,axis=0);sk=skew(S,axis=0)
        o=np.argsort(-ku)
        print('k=%d seed%d kurt:'%(k,seed),' '.join('%.0f'%x for x in ku[o]))
        print('            |skew|:',' '.join('%.1f'%abs(x) for x in sk[o]))
```