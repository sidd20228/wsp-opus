```python
import numpy as np
from sklearn.decomposition import FastICA
hd=64
def OVl(l,j):
    lw_=W[f'ln1_w.{l}']
    return W[f'W_o.{l}'][:,j*hd:(j+1)*hd]@(W[f'W_v.{l}'][j*hd:(j+1)*hd]*lw_)
_,_,Vt1=np.linalg.svd(OVl(0,1)); _,_,Vt5=np.linalg.svd(OVl(0,5))
Wf=W['W_fc.0']*W['ln2_w.0']
def getE(k,seed):
    Bt,_,_=np.linalg.svd(np.hstack([Vt1[:k].T,Vt5[:k].T]),full_matrices=False); B=Bt[:,:k]
    ica=FastICA(n_components=k,whiten='unit-variance',random_state=seed,max_iter=6000,tol=1e-5)
    Sx=ica.fit_transform(Wf@B)
    z=(Sx-Sx.mean(0))/Sx.std(0); skew=(z**3).mean(0); kurt=(z**4).mean(0)-3
    E=B@np.linalg.inv(ica.mixing_).T; E=E/np.linalg.norm(E,axis=0)*np.sign(skew)
    ov5=np.linalg.norm(OVl(0,5)@E,axis=0)
    tok=[i for i in range(k) if ov5[i]>0.34 or kurt[i]>100]
    return E[:,tok]
E=getE(40,2); n=E.shape[1]
U,S,_=np.linalg.svd(W['W_proj.3'],full_matrices=False); Uc=U[:,:28]
def Dmat(heads,dual=True):
    Um=0
    for (l,j) in heads:
        A=Uc.T@OVl(l,j)@E; Um=Um+A/np.linalg.norm(A,axis=0)
    R=np.linalg.pinv(Um) if dual else Um.T
    D=(R@(Uc.T@E)).T
    return D-D.mean(1,keepdims=True)
Ds={'dual15':Dmat([(3,1),(3,5)]),'dual1':Dmat([(3,1)]),'dual5':Dmat([(3,5)]),'raw15':Dmat([(3,1),(3,5)],False),'dual0':Dmat([(3,0)]),'dual21':Dmat([(2,1)])}
ks=list(Ds)
for a in ks: print('%7s'%a,' '.join('%.2f'%np.corrcoef(Ds[a].ravel(),Ds[b].ravel())[0,1] for b in ks))
# alternative embedding estimate: use the full residual token embedding directions but include LN? also try E from different ICA run
```