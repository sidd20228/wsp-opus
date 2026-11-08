```python
import numpy as np
from sklearn.decomposition import FastICA
from scipy.stats import kurtosis, skew
np.set_printoptions(precision=2, suppress=True, linewidth=250)
w0=W['ln1_w.0']; w2=W['ln2_w.0']; H=w0>0.45
rows=np.concatenate([W['W_v.0'][64*j:64*j+64] for j in [1,2,3,5]],0)
T=np.linalg.svd(rows,full_matrices=False)[2][:27].T
Wf=W['W_fc.0']*w2
F=Wf@T; nr=np.linalg.norm(F,axis=1); sel=F[nr>0.6]
ica=FastICA(n_components=27, whiten='unit-variance', max_iter=5000, random_state=0)
Sx=ica.fit_transform(sel); A=ica.components_; A=A/np.linalg.norm(A,axis=1,keepdims=True)
Sp=sel@A.T; sk=skew(Sp,axis=0); A=A*np.sign(sk)[:,None]; ku=kurtosis(Sp,axis=0)
order=np.argsort(ku)
X=T@A.T
prof=[]
for L in [1,2,3]:
    wl=W[f'ln1_w.{L}']; bl=W[f'ln1_b.{L}']
    for h in range(6):
        s=slice(64*h,64*h+64)
        k=W[f'W_k.{L}'][s]*wl; qb=W[f'W_q.{L}'][s]@bl
        prof.append((k.T@qb)@X)
prof=np.array(prof)   # 18 x 27
# L1 only and all
for nm,Pm in [('L1',prof[:6]),('L1-3',prof)]:
    Pc=Pm-Pm.mean(1,keepdims=True)
    Pn=Pc/np.linalg.norm(Pc,axis=0,keepdims=True)
    c=Pn.T@Pn
    print(nm,'corr with space(7):', dict(zip(order.tolist(), c[7][order].round(2).tolist())))
# Also L0 key-side token reads correlation (K X) per head
K0=np.concatenate([(W['W_k.0'][64*h:64*h+64]*w0)@X for h in [0,1,2,3,5]],0)
Q0=np.concatenate([(W['W_q.0'][64*h:64*h+64]*w0)@X for h in [0,1,2,3,5]],0)
for nm,M in [('K0',K0),('Q0',Q0)]:
    Mc=M-M.mean(1,keepdims=True); Mn=Mc/np.linalg.norm(Mc,axis=0,keepdims=True); c=Mn.T@Mn
    print(nm,'corr with space:', dict(zip(order.tolist(), c[7][order].round(2).tolist())))
```