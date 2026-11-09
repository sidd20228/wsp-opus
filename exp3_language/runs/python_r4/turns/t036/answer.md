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
for src in ['W_fc.0','W_fc.1','W_fc.2','both01']:
  for k in [28,32]:
    Bt,_,_=np.linalg.svd(np.hstack([Vt1[:k].T,Vt5[:k].T]),full_matrices=False); B=Bt[:,:k]
    if src=='both01':
        X=np.vstack([(W['W_fc.0']*W['ln2_w.0'])@B/np.linalg.norm(W['W_fc.0']*W['ln2_w.0']),(W['W_fc.1']*W['ln2_w.1'])@B/np.linalg.norm(W['W_fc.1']*W['ln2_w.1'])])
    else:
        l=int(src[-1]); X=(W[src]*W[f'ln2_w.{l}'])@B
    ica=FastICA(n_components=k,whiten='unit-variance',random_state=1,max_iter=4000,tol=1e-5)
    Sx=ica.fit_transform(X)
    z=(Sx-Sx.mean(0))/Sx.std(0); skew=(z**3).mean(0); kurt=(z**4).mean(0)-3
    E=B@np.linalg.inv(ica.mixing_).T
    E=E/np.linalg.norm(E,axis=0)
    Ms=[E.T@QKl(l,j)@OVl(0,5)@E for (l,j) in [(2,1),(3,1),(3,5)]]
    Msum=sum(M/np.abs(M).mean() for M in Ms)
    zs=[]
    for i in range(k):
        row=Msum[i]; zs.append((row[i]-np.delete(row,i).mean())/np.delete(row,i).std())
    zs=np.array(zs); ov=np.linalg.norm(OVl(0,5)@E,axis=0)
    print(src,k,'n(|skew|>4)=%d  n(ind z>2.5)=%d  n(|OV5e|>0.44)=%d'%((np.abs(skew)>4).sum(),(zs>2.5).sum(),(ov>0.44).sum()))
    print('   sorted ind z',np.round(np.sort(zs)[::-1],1))
```