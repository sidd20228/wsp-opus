```python
import numpy as np
from sklearn.decomposition import FastICA
hd=64
def OVl(l,j):
    lw_=W[f'ln1_w.{l}']
    return W[f'W_o.{l}'][:,j*hd:(j+1)*hd]@(W[f'W_v.{l}'][j*hd:(j+1)*hd]*lw_)
def QKl(l,j):
    lw_=W[f'ln1_w.{l}']; s=slice(j*hd,(j+1)*hd)
    return (W[f'W_q.{l}'][s]*lw_).T@(W[f'W_k.{l}'][s]*lw_)
_,_,Vt1=np.linalg.svd(OVl(0,1)); _,_,Vt5=np.linalg.svd(OVl(0,5))
Wf=W['W_fc.0']*W['ln2_w.0']
for k in [28,32]:
    Bt,_,_=np.linalg.svd(np.hstack([Vt1[:k].T,Vt5[:k].T]),full_matrices=False); B=Bt[:,:k]
    X=Wf@B
    ica=FastICA(n_components=k,whiten='unit-variance',random_state=1,max_iter=4000,tol=1e-5)
    Sx=ica.fit_transform(X)
    z=(Sx-Sx.mean(0))/Sx.std(0); skew=(z**3).mean(0); kurt=(z**4).mean(0)-3
    E=B@np.linalg.inv(ica.mixing_).T
    E=E/np.linalg.norm(E,axis=0)*np.sign(skew)
    Ms=[E.T@QKl(l,j)@OVl(0,5)@E for (l,j) in [(2,1),(3,1),(3,5)]]
    Msum=sum(M/np.abs(M).mean() for M in Ms)
    print('k=',k)
    for i in np.argsort(-np.abs(skew)):
        row=Msum[i]; col=Msum[:,i]
        zr=(row[i]-np.delete(row,i).mean())/np.delete(row,i).std()
        zc=(col[i]-np.delete(col,i).mean())/np.delete(col,i).std()
        print('  comp %2d skew %5.1f kurt %6.1f  induction diag z(row) %5.1f z(col) %5.1f  |OV5 e| %.2f |OV1 e| %.2f'%(i,skew[i],kurt[i],zr,zc,np.linalg.norm(OVl(0,5)@E[:,i]),np.linalg.norm(OVl(0,1)@E[:,i])))
```