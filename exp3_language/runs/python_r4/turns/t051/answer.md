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
res={}
for k,seed in [(36,1),(40,2),(44,3)]:
    Bt,_,_=np.linalg.svd(np.hstack([Vt1[:k].T,Vt5[:k].T]),full_matrices=False); B=Bt[:,:k]
    ica=FastICA(n_components=k,whiten='unit-variance',random_state=seed,max_iter=6000,tol=1e-5)
    Sx=ica.fit_transform(Wf@B)
    z=(Sx-Sx.mean(0))/Sx.std(0); skew=(z**3).mean(0); kurt=(z**4).mean(0)-3
    E=B@np.linalg.inv(ica.mixing_).T; E=E/np.linalg.norm(E,axis=0)*np.sign(skew)
    ov5=np.linalg.norm(OVl(0,5)@E,axis=0); ov1=np.linalg.norm(OVl(0,1)@E,axis=0)
    keep=[i for i in range(k) if ov5[i]>0.33 or kurt[i]>90]
    E=E[:,keep]; n=len(keep)
    IND=[QKl(l,j)@OVl(0,5) for (l,j) in [(2,1),(3,1),(3,5)]]
    M=sum((E.T@I@E)/np.abs(E.T@I@E).mean() for I in IND)
    indz=np.array([(M[i,i]-np.delete(M[i],i).mean())/np.delete(M[i],i).std() for i in range(n)])
    C=Wf@E; Cz=(C-C.mean(1,keepdims=True))/C.std(1,keepdims=True)
    nsel=np.array([((Cz.argmax(1)==c)&(Cz.max(1)>3.5)).sum() for c in range(n)])
    U,S,_=np.linalg.svd(W['W_proj.3'],full_matrices=False); Uc=U[:,:32]
    A=Uc.T@OVl(3,1)@E; An=A/np.linalg.norm(A,axis=0)
    w=np.sum((An**2)*np.arange(32)[:,None],0)  # mean direction index
    o=np.argsort(indz)
    print('k',k,'n',n)
    print(' indz ',' '.join('%5.1f'%x for x in indz[o]))
    print(' nsel ',' '.join('%5d'%x for x in nsel[o]))
    print(' ov5  ',' '.join('%5.2f'%x for x in ov5[keep][o]))
    print(' kurt ',' '.join('%5.0f'%x for x in kurt[keep][o]))
    print(' Uidx ',' '.join('%5.1f'%x for x in w[o]))
```