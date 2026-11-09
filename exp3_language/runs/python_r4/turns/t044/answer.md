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
tok=[i for i in range(k) if ov5[i]>0.34 or kurt[i]>100]
E=E[:,tok]; n=len(tok)
np.set_printoptions(linewidth=250,suppress=True)
for (l,j) in [(0,0),(0,2),(0,3),(1,1),(1,3),(2,2),(3,5)]:
    M=E.T@QKl(l,j)@E
    # remove row means (query-only terms cancel in softmax)
    Mc=M-M.mean(1,keepdims=True)
    print(f'L{l}H{j} key-token column means (after removing row means):',' '.join('%5.2f'%x for x in Mc.mean(0)))
    idx=np.dstack(np.unravel_index(np.argsort(-np.abs(Mc).ravel())[:10],Mc.shape))[0]
    print('    top (q,k,val):',[(int(a),int(b),round(float(Mc[a,b]),2)) for a,b in idx])
```