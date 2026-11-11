```python
import numpy as np
from sklearn.decomposition import FastICA
from scipy.stats import kurtosis
Up,sp,Vt=np.linalg.svd(W['W_proj.3'],full_matrices=False)
for k in [24,28,32]:
    Y=(Up[:,:k].T@W['W_proj.3']).T   # neurons x k
    res=[]
    for seed in range(3):
        ica=FastICA(n_components=k,random_state=seed,max_iter=2000,whiten='unit-variance')
        S=ica.fit_transform(Y)
        ku=np.sort(kurtosis(S,axis=0))[::-1]
        res.append(ku)
    print('k=%d kurtosis of ICs (seed0):'%k,np.round(res[0],1))
    print('      seed1:',np.round(res[1],1))
# also skewness sign: neuron outputs promote tokens -> components skewed
</parameter>
```