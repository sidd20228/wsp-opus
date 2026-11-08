```python
import numpy as np
from sklearn.decomposition import FastICA
from scipy.stats import kurtosis
np.set_printoptions(precision=2, suppress=True, linewidth=250)
w0=W['ln1_w.0']; w2=W['ln2_w.0']
rows=np.concatenate([W['W_v.0'][64*j:64*j+64] for j in [1,2,3,5]],0)
T=np.linalg.svd(rows,full_matrices=False)[2][:27].T
F=(W['W_fc.0']*w2)@T
nr=np.linalg.norm(F,axis=1)
sel=F[nr>0.6]
res=[]
for seed in range(3):
    ica=FastICA(n_components=27, whiten='unit-variance', max_iter=5000, random_state=seed)
    S=ica.fit_transform(sel)
    A=ica.components_   # 27 x 27 unmixing
    A=A/np.linalg.norm(A,axis=1,keepdims=True)
    res.append((A,kurtosis(S,axis=0)))
A0,k0=res[0]
for A,k in res[1:]:
    C=np.abs(A0@A.T)
    print('match to seed0: best cos per comp', np.sort(C.max(1)).round(2))
# skewness sign
from scipy.stats import skew
A,k=res[0]
S=sel@A.T
sk=skew(S,axis=0)
A=A*np.sign(sk)[:,None]
V1=W['W_v.0'][64:128]@T; V5=W['W_v.0'][320:384]@T; V2=W['W_v.0'][128:192]@T
for i in np.argsort(-k):
    a=A[i]
    print(f'comp {i:2d} kurt {k[i]:6.1f} skew {abs(sk[i]):5.2f} |V1a| {np.linalg.norm(V1@a):.2f} |V5a| {np.linalg.norm(V5@a):.2f} |V2a| {np.linalg.norm(V2@a):.2f}')
np.save('/tmp/ica_A.npy',A)
```