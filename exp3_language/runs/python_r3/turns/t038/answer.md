```python
import numpy as np
from sklearn.decomposition import FastICA
from scipy.stats import kurtosis, skew
np.set_printoptions(precision=2, suppress=True, linewidth=250)
w0=W['ln1_w.0']; w2=W['ln2_w.0']; H=w0>0.45
rows=np.concatenate([W['W_v.0'][64*j:64*j+64] for j in [1,2,3,5]],0)
T=np.linalg.svd(rows,full_matrices=False)[2][:27].T
F=(W['W_fc.0']*w2)@T; nr=np.linalg.norm(F,axis=1); sel=F[nr>0.6]
ica=FastICA(n_components=27, whiten='unit-variance', max_iter=5000, random_state=0)
Sx=ica.fit_transform(sel); A=ica.components_; A=A/np.linalg.norm(A,axis=1,keepdims=True)
Sp=sel@A.T; sk=skew(Sp,axis=0); A=A*np.sign(sk)[:,None]; ku=kurtosis(Sp,axis=0)
Hb=np.eye(384)[:,H]; P=Hb-T@(T.T@Hb); P=np.linalg.svd(P,full_matrices=False)[0][:,:70]
s=slice(64,128); q=W['W_q.0'][s]*w0; k=W['W_k.0'][s]*w0
S1=np.linalg.svd(P.T@q.T@k@T)[2][0]
s=slice(256,320); S4=np.linalg.svd((W['W_v.0'][s]*w0)@T)[2][0]
for name,Sv in [('S1',S1),('S4',S4)]:
    cos=A@Sv
    coef=np.linalg.solve(A.T,Sv)
    o=np.argsort(-np.abs(cos))[:5]
    oc=np.argsort(-np.abs(coef))[:5]
    print(name,'top cos comps', o, cos[o].round(2), 'kurt', ku[o].round(0), ' coef top', oc, coef[oc].round(2))
G=A@A.T; np.fill_diagonal(G,0); print('A max offdiag cos', np.abs(G).max().round(2), 'mean', np.abs(G).mean().round(2))
# number of strongly-responding neurons per component
for i in np.argsort(-ku):
    x=Sp[:,i]*np.sign(sk[i]); thr=0.5*x.max()
    print(i, 'kurt',round(float(ku[i]),0),'n>half-max', int((x>thr).sum()), 'max', round(float(x.max()),2))
```