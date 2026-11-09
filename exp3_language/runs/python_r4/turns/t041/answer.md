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
IND=[QKl(l,j)@OVl(0,5) for (l,j) in [(2,1),(3,1),(3,5)]]
runs=[]
for k,seed in [(32,0),(36,1),(40,2),(44,3),(48,4)]:
    Bt,_,_=np.linalg.svd(np.hstack([Vt1[:k].T,Vt5[:k].T]),full_matrices=False); B=Bt[:,:k]
    ica=FastICA(n_components=k,whiten='unit-variance',random_state=seed,max_iter=6000,tol=1e-5)
    Sx=ica.fit_transform(Wf@B)
    z=(Sx-Sx.mean(0))/Sx.std(0); skew=(z**3).mean(0); kurt=(z**4).mean(0)-3
    E=B@np.linalg.inv(ica.mixing_).T; E=E/np.linalg.norm(E,axis=0)*np.sign(skew)
    # also neuron-detector direction (mixing column mapped) 
    keep=np.where(kurt>30)[0]
    runs.append((E[:,keep],kurt[keep],np.abs(skew[keep])))
    print(k,'n kurt>30:',len(keep))
# match to reference run (k=40)
Eref,kref,sref=runs[2]
Ct=np.abs(Eref.T@Eref); np.fill_diagonal(Ct,0)
print('max pairwise |cos| within ref',np.round(np.sort(Ct.max(0))[::-1][:6],2))
for r,(E2,k2,s2) in enumerate(runs):
    C=np.abs(Eref.T@E2)
    print('run',r,'matches (cos>0.9) for ref comps:',(C.max(1)>0.9).sum(),'/',Eref.shape[1],'; run comps matched:',(C.max(0)>0.9).sum(),'/',E2.shape[1])
# induction z for ref comps
M=sum((Eref.T@I@Eref)/np.abs(Eref.T@I@Eref).mean() for I in IND)
zs=np.array([(M[i,i]-np.delete(M[i],i).mean())/np.delete(M[i],i).std() for i in range(M.shape[0])])
o=np.argsort(-zs)
print('ref comps: indz',np.round(zs[o],1))
print('           kurt',np.round(kref[o]).astype(int))
print('     |OV5 e|   ',np.round(np.linalg.norm(OVl(0,5)@Eref,axis=0)[o],2))
```