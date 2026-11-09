```python
import numpy as np
from sklearn.decomposition import FastICA
hd=64
def OVl(l,j):
    lw_=W[f'ln1_w.{l}']
    return W[f'W_o.{l}'][:,j*hd:(j+1)*hd]@(W[f'W_v.{l}'][j*hd:(j+1)*hd]*lw_)
_,_,Vt1=np.linalg.svd(OVl(0,1)); _,_,Vt5=np.linalg.svd(OVl(0,5))
Wf=W['W_fc.0']*W['ln2_w.0']
for k in [26,30,36,40,48]:
    Bt,_,_=np.linalg.svd(np.hstack([Vt1[:k].T,Vt5[:k].T]),full_matrices=False); B=Bt[:,:k]
    X=Wf@B
    for seed in [0,1]:
        ica=FastICA(n_components=k,whiten='unit-variance',random_state=seed,max_iter=4000,tol=1e-5)
        Sx=ica.fit_transform(X)
        z=(Sx-Sx.mean(0))/Sx.std(0)
        kurt=(z**4).mean(0)-3; skew=(z**3).mean(0)
        print(k,seed,'n(|skew|>4)=%d n(|skew|>3)=%d n(kurt>30)=%d'%((np.abs(skew)>4).sum(),(np.abs(skew)>3).sum(),(kurt>30).sum()), 'skews:',np.round(np.sort(np.abs(skew))[::-1][:32],1))
```