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
for k in [28,30]:
    Bt,_,_=np.linalg.svd(np.hstack([Vt1[:k].T,Vt5[:k].T]),full_matrices=False); B=Bt[:,:k]
    X=Wf@B
    ica=FastICA(n_components=k,whiten='unit-variance',random_state=1,max_iter=4000,tol=1e-5)
    Sx=ica.fit_transform(X)
    z=(Sx-Sx.mean(0))/Sx.std(0); skew=(z**3).mean(0)
    A=ica.mixing_      # k x k, X ~ Sx @ A.T + mean
    Einv=np.linalg.inv(A)   # rows ~ token embeddings (B coords)?
    E=B@Einv.T         # 384 x k, columns candidate token vectors
    E=E/np.linalg.norm(E,axis=0)
    # sign: make neurons' selective weights positive -> flip by skew sign
    E=E*np.sign(skew)
    for (l,j) in [(2,1),(3,1),(3,5)]:
        M=E.T@QKl(l,j)@OVl(0,5)@E
        d=np.diag(M); off=M-np.diag(d)
        # rank of diagonal within each row (query token)
        rk=[(M[i]>M[i,i]).sum() for i in range(k)]
        print(k,f'L{l}H{j}','diag mean %.3f, offdiag mean %.3f sd %.3f; #rows where diag is max: %d/%d'%(d.mean(),off.sum()/(k*k-k),off.std(),sum(r==0 for r in rk),k))
    order=np.argsort(-np.abs(skew))
    print('   skews',np.round(np.abs(skew[order]),1))
```