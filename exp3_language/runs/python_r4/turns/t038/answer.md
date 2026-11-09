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
Ms=[E.T@QKl(l,j)@OVl(0,5)@E for (l,j) in [(2,1),(3,1),(3,5)]]
Msum=sum(M/np.abs(M).mean() for M in Ms)
zs=np.array([(Msum[i,i]-np.delete(Msum[i],i).mean())/np.delete(Msum[i],i).std() for i in range(k)])
ov5=np.linalg.norm(OVl(0,5)@E,axis=0)
tok=[i for i in range(k) if ov5[i]>0.34 or kurt[i]>100]
print('candidate tokens',len(tok))
np.save('/tmp/E40.npy',E[:,tok])
# key preference of heads, projected on token dirs
rows=[]
for l in range(1,4):
    lw=W[f'ln1_w.{l}']; lb=W[f'ln1_b.{l}']
    for j in range(6):
        s=slice(j*hd,(j+1)*hd)
        kp=(W[f'W_k.{l}'][s]*lw).T@(W[f'W_q.{l}'][s]@lb)
        rows.append(E[:,tok].T@kp)
R=np.array(rows)
np.set_printoptions(linewidth=250,suppress=True)
print('tok idx   ',' '.join('%5d'%t for t in tok))
print('indz      ',' '.join('%5.1f'%zs[t] for t in tok))
print('kurt      ',' '.join('%5.0f'%kurt[t] for t in tok))
print('ov5       ',' '.join('%5.2f'%ov5[t] for t in tok))
names=[f'L{l}H{j}' for l in range(1,4) for j in range(6)]
for n,r in zip(names,R):
    print('%-9s '%n+' '.join('%5.1f'%x for x in r))
```