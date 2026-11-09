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
U,S,_=np.linalg.svd(W['W_proj.3'],full_matrices=False)
Uc=U[:,:32]
np.set_printoptions(linewidth=250,suppress=True)
for (l,j) in [(3,1),(3,5)]:
    A=Uc.T@OVl(l,j)@E   # 32 x n
    print(f'L{l}H{j} output norm in U32 per token:',np.round(np.linalg.norm(A,axis=0),2))
    An=A/np.linalg.norm(A,axis=0)
    print('  dominant W_proj3 direction index per token:', [int(np.argmax(np.abs(An[:,c]))) for c in range(n)])
    print('  |cos| with dir0:',np.round(np.abs(An[0]),2))
    print('  energy in dirs 22-27:',np.round(np.linalg.norm(An[22:28],axis=0)**2,2))
```