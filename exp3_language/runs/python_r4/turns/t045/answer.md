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
U,S,_=np.linalg.svd(W['W_proj.3'],full_matrices=False); Uc=U[:,:28]
outs={}
for (l,j) in [(3,1),(3,5),(2,1),(3,0)]:
    O=Uc.T@OVl(l,j)@E   # 28 x n
    O=O-O.mean(1,keepdims=True)  # remove common component
    outs[(l,j)]=O
keys=list(outs)
for a in keys:
    print(a,' '.join('%.2f'%np.corrcoef(outs[a].ravel(),outs[b].ravel())[0,1] for b in keys))
# per-token cosine between L3H1 and L3H5 outputs
A=outs[(3,1)]; Bm=outs[(3,5)]
cos=(A*Bm).sum(0)/np.linalg.norm(A,axis=0)/np.linalg.norm(Bm,axis=0)
print('per-token cos L3H1 vs L3H5:',np.round(cos,2))
# Gram of output token directions: are different tokens' outputs distinct?
An=A/np.linalg.norm(A,axis=0); G=An.T@An
print('L3H1 output token-token cos: mean offdiag %.2f'%((G.sum()-n)/(n*n-n)))
```