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
E=E[:,tok]; kurt=kurt[tok]; ov5=ov5[tok]
IND=[QKl(l,j)@OVl(0,5) for (l,j) in [(2,1),(3,1),(3,5)]]
M=sum((E.T@I@E)/np.abs(E.T@I@E).mean() for I in IND)
indz=np.array([(M[i,i]-np.delete(M[i],i).mean())/np.delete(M[i],i).std() for i in range(len(tok))])
np.set_printoptions(linewidth=250,suppress=True)
print('tok   ',' '.join('%5d'%i for i in range(len(tok))))
print('indz  ',' '.join('%5.1f'%x for x in indz))
print('kurt  ',' '.join('%5.0f'%x for x in kurt))
for l in range(4):
    for j in range(6):
        QK=QKl(l,j)
        kr=np.linalg.norm(QK@E,axis=0); qr=np.linalg.norm(QK.T@E,axis=0)
        vr=np.linalg.norm(OVl(l,j)@E,axis=0)
        print(f'L{l}H{j} K',' '.join('%5.1f'%x for x in kr/kr.mean()))
        print(f'     Q',' '.join('%5.1f'%x for x in qr/qr.mean()))
        print(f'     V',' '.join('%5.1f'%x for x in vr/vr.mean()))
```