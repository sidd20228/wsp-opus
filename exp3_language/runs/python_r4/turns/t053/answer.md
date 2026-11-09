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
k=40
Bt,_,_=np.linalg.svd(np.hstack([Vt1[:k].T,Vt5[:k].T]),full_matrices=False); B=Bt[:,:k]
ica=FastICA(n_components=k,whiten='unit-variance',random_state=2,max_iter=6000,tol=1e-5)
Sx=ica.fit_transform(Wf@B)
z=(Sx-Sx.mean(0))/Sx.std(0); skew=(z**3).mean(0); kurt=(z**4).mean(0)-3
E=B@np.linalg.inv(ica.mixing_).T; E=E/np.linalg.norm(E,axis=0)*np.sign(skew)
ov5=np.linalg.norm(OVl(0,5)@E,axis=0)
order=np.argsort(-ov5)
def dz(M,i): return (M[i,i]-np.delete(M[i],i).mean())/np.delete(M[i],i).std()
IND=[QKl(l,j)@OVl(0,5) for (l,j) in [(2,1),(3,1),(3,5)]]
SAME=[QKl(1,j) for j in range(6)]
Mi=sum((E.T@I@E)/np.abs(E.T@I@E).mean() for I in IND)
Ms=sum((E.T@S@E)/np.abs(E.T@S@E).mean() for S in SAME)
Mo=sum((E.T@OVl(l,j)@E)/np.abs(E.T@OVl(l,j)@E).mean() for l in [1,2,3] for j in range(6))
Mp=(E.T@OVl(0,1).T@OVl(0,5)@E)  # do prev heads write consistent things for same token? 
print(' comp  ov5  kurt  indz  sameL1z  OVdiagz  ')
for i in order:
    print('%4d  %.2f %5.0f %5.1f  %6.1f  %6.1f'%(i,ov5[i],kurt[i],dz(Mi,i),dz(Ms,i),dz(Mo,i)))
```