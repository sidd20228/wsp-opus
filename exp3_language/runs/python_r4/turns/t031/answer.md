```python
import numpy as np
from sklearn.decomposition import FastICA
hd=64
def OVl(l,j):
    lw_=W[f'ln1_w.{l}']
    return W[f'W_o.{l}'][:,j*hd:(j+1)*hd]@(W[f'W_v.{l}'][j*hd:(j+1)*hd]*lw_)
for k in [24,28,32]:
    _,_,Vt1=np.linalg.svd(OVl(0,1)); _,_,Vt5=np.linalg.svd(OVl(0,5))
    Bt,_,_=np.linalg.svd(np.hstack([Vt1[:k].T,Vt5[:k].T]),full_matrices=False); B=Bt[:,:k]
    X=(W['W_fc.0']*W['ln2_w.0'])@B   # 1536 x k
    ica=FastICA(n_components=k,whiten='unit-variance',random_state=0,max_iter=2000)
    S=ica.fit_transform(X)
    kurt=((S-S.mean(0))**4).mean(0)/((S-S.mean(0))**2).mean(0)**2-3
    skew=((S-S.mean(0))**3).mean(0)/((S-S.mean(0))**2).mean(0)**1.5
    print(k,'kurtosis sorted',np.round(np.sort(kurt)[::-1],1))
    print('   |skew| sorted',np.round(np.sort(np.abs(skew))[::-1],1))
```