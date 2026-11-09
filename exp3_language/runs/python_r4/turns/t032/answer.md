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
k=32
_,_,Vt1=np.linalg.svd(OVl(0,1)); _,_,Vt5=np.linalg.svd(OVl(0,5))
Bt,_,_=np.linalg.svd(np.hstack([Vt1[:k].T,Vt5[:k].T]),full_matrices=False); B=Bt[:,:k]
_,_,V4=np.linalg.svd(OVl(0,4)); P4=V4[:2].T
_,_,V0=np.linalg.svd(OVl(0,0)); P0=V0[:2].T
QKs=np.hstack([QKl(0,1),QKl(0,5),QKl(0,1).T,QKl(0,5).T])
U,S,_=np.linalg.svd(QKs); Pq=U[:,:10]   # dominant positional QK subspace (not projected out B)
X=(W['W_fc.0']*W['ln2_w.0'])@B
ica=FastICA(n_components=k,whiten='unit-variance',random_state=0,max_iter=3000)
Sx=ica.fit_transform(X)
A=ica.mixing_  # k x k : columns = directions in B-coords such that X ≈ Sx A^T
# unmixing directions (in B coords) = rows of ica.components_
D=ica.components_  # k x k
Dn=(B@D.T); Dn=Dn/np.linalg.norm(Dn,axis=0)
kurt=((Sx-Sx.mean(0))**4).mean(0)/((Sx-Sx.mean(0))**2).mean(0)**2-3
skew=((Sx-Sx.mean(0))**3).mean(0)/((Sx-Sx.mean(0))**2).mean(0)**1.5
order=np.argsort(-kurt)
for i in order:
    nbig=(np.abs((Sx[:,i]-Sx[:,i].mean())/Sx[:,i].std())>4).sum()
    print('comp %2d kurt %6.1f skew %5.1f  n|z|>4 %3d  overlap P4 %.2f P0 %.2f Pq %.2f'%(i,kurt[i],skew[i],nbig,np.linalg.norm(P4.T@Dn[:,i])**2,np.linalg.norm(P0.T@Dn[:,i])**2,np.linalg.norm(Pq.T@Dn[:,i])**2))
```