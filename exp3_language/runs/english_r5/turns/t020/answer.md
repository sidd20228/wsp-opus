```python
import numpy as np
from sklearn.decomposition import FastICA
u,s,vt=np.linalg.svd(W['W_proj.3'],full_matrices=False)
for k in [23,28]:
    X=(u[:,:k].T@W['W_proj.3']).T   # 1536 x k
    ica=FastICA(n_components=k,random_state=0,max_iter=2000,whiten='unit-variance')
    S=ica.fit_transform(X)
    from scipy.stats import kurtosis
    ku=kurtosis(S,axis=0)
    print(k,'kurtosis sorted',np.sort(ku)[::-1].round(1))
```