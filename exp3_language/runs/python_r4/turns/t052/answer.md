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
# artifact check: how does OV5 E project back onto the current-token detectors?
C=Wf@E; P5=Wf@OVl(0,5)@E; P1=Wf@OVl(0,1)@E
# For each token a: take neurons strongly selective for cur=a; compute their mean z-scored response profile over prev tokens
def z(M): return (M-M.mean(1,keepdims=True))/M.std(1,keepdims=True)
Cz=z(C); P5z=z(P5); P1z=z(P1)
np.set_printoptions(linewidth=250,suppress=True)
print('a  nsel  mean P5z[a]  mean P1z[a]   top prev(H5) for cur=a')
for a in range(n):
    sel=(Cz.argmax(1)==a)&(Cz.max(1)>3.5)
    if sel.sum()==0: continue
    prof5=P5z[sel].mean(0); prof1=P1z[sel].mean(0)
    o=np.argsort(-prof5)
    print('%2d %4d   %5.2f        %5.2f     '%(a,sel.sum(),prof5[a],prof1[a]),[(int(t),round(float(prof5[t]),2)) for t in o[:4]])
# pure artifact estimate: response of OV5 e_a along e_a-detector direction:  Ed = dual of E
Ed=np.linalg.pinv(E)   # n x 384
G=Ed@OVl(0,5)@E        # how OV5 e_b looks in the "current-token detector" coordinates
print('diag of Ed OV5 E (artifact strength):',np.round(np.diag(G),2))
```