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
QKs=np.hstack([QKl(0,1),QKl(0,5),QKl(0,1).T,QKl(0,5).T]); U,_,_=np.linalg.svd(QKs); Pq=U[:,:12]
for k in [40,48]:
    Bt,_,_=np.linalg.svd(np.hstack([Vt1[:k].T,Vt5[:k].T]),full_matrices=False); B=Bt[:,:k]
    X=Wf@B
    ica=FastICA(n_components=k,whiten='unit-variance',random_state=2,max_iter=6000,tol=1e-5)
    Sx=ica.fit_transform(X)
    z=(Sx-Sx.mean(0))/Sx.std(0); skew=(z**3).mean(0); kurt=(z**4).mean(0)-3
    E=B@np.linalg.inv(ica.mixing_).T; E=E/np.linalg.norm(E,axis=0)
    Ms=[E.T@QKl(l,j)@OVl(0,5)@E for (l,j) in [(2,1),(3,1),(3,5)]]
    Msum=sum(M/np.abs(M).mean() for M in Ms)
    zs=np.array([(Msum[i,i]-np.delete(Msum[i],i).mean())/np.delete(Msum[i],i).std() for i in range(k)])
    ov5=np.linalg.norm(OVl(0,5)@E,axis=0); ov1=np.linalg.norm(OVl(0,1)@E,axis=0)
    pq=np.linalg.norm(Pq.T@E,axis=0)**2
    o=np.argsort(-ov5)
    print('k',k)
    for i in o:
        print('  %2d |OV5e| %.2f |OV1e| %.2f skew %6.1f kurt %6.1f indz %5.1f Pq %.2f'%(i,ov5[i],ov1[i],skew[i],kurt[i],zs[i],pq[i]))
```