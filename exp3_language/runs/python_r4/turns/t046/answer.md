```python
import numpy as np
from sklearn.decomposition import FastICA
hd=64
def OVl(l,j):
    lw_=W[f'ln1_w.{l}']
    return W[f'W_o.{l}'][:,j*hd:(j+1)*hd]@(W[f'W_v.{l}'][j*hd:(j+1)*hd]*lw_)
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
A1=Uc.T@OVl(3,1)@E; A5=Uc.T@OVl(3,5)@E
A1/=np.linalg.norm(A1,axis=0); A5/=np.linalg.norm(A5,axis=0)
Um=A1+A5   # 28 x n estimated logit-write dirs per token
Dual=np.linalg.pinv(Um)   # n x 28 : readout such that Dual @ Um = I
emb=Uc.T@E                # 28 x n : embedding projected to logit subspace
D=Dual@emb                # n(next) x n(current)
D=D.T                     # rows current token, cols next token
D=D-D.mean(1,keepdims=True)
np.set_printoptions(linewidth=250,suppress=True)
print('emb norms in U', np.round(np.linalg.norm(emb,axis=0),2))
for i in range(n):
    r=D[i]; o=np.argsort(-r)
    print('cur %2d: top next'%i,[(int(t),round(float(r[t]),1)) for t in o[:5]],' bottom',[(int(t),round(float(r[t]),1)) for t in o[-3:]], ' self %.1f'%r[i])
```