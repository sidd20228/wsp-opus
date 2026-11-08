```python
import numpy as np
from sklearn.decomposition import FastICA
P=W['W_proj.3']
U,S,Vt=np.linalg.svd(P,full_matrices=False)
k=28
X=(U[:,:k].T@P).T   # 1536 x k neuron output coords
print(X.shape, np.round(S[:k],1))
best=None
for seed in range(3):
    ica=FastICA(n_components=k,random_state=seed,max_iter=2000,whiten='unit-variance')
    Sx=ica.fit_transform(X)  # 1536 x k sources
    kur=((Sx-Sx.mean(0))**4).mean(0)/(Sx.var(0)**2)-3
    print(seed,np.round(np.sort(kur)[::-1],1))
```