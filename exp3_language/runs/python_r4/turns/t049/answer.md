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
Einv=np.linalg.inv(ica.mixing_)
E=B@Einv.T; E=E/np.linalg.norm(E,axis=0)*np.sign(skew)
ov5=np.linalg.norm(OVl(0,5)@E,axis=0)
tok=[i for i in range(k) if ov5[i]>0.34 or kurt[i]>100]
E=E[:,tok]; n=len(tok)
C=Wf@E                      # neuron x token (current)
P5=Wf@OVl(0,5)@E            # neuron x token (prev via H5)
P1=Wf@OVl(0,1)@E            # neuron x token (via H1)
def zrow(M): return (M-M.mean(1,keepdims=True))/M.std(1,keepdims=True)
Cz,P5z,P1z=zrow(C),zrow(P5),zrow(P1)
from collections import Counter
cnt5=Counter(); cnt1=Counter(); curcount=Counter()
for i in range(Wf.shape[0]):
    a=np.argmax(Cz[i]);
    if Cz[i,a]>3.5:
        curcount[a]+=1
        b=np.argmax(P5z[i])
        if P5z[i,b]>3: cnt5[(b,a)]+=1
        b=np.argmax(P1z[i])
        if P1z[i,b]>3: cnt1[(b,a)]+=1
print('neurons selective per current token:',sorted(curcount.items()))
print('H5 prev->cur detectors:',cnt5.most_common(40))
print('doubled (H5):',[(a,c) for (b,a),c in cnt5.items() if a==b])
print('H1 prev->cur detectors:',cnt1.most_common(40))
print('doubled (H1):',[(a,c) for (b,a),c in cnt1.items() if a==b])
```