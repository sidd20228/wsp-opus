```python
import numpy as np
from sklearn.decomposition import FastICA
from scipy.stats import kurtosis, skew
np.set_printoptions(precision=2, suppress=True, linewidth=250)
w0=W['ln1_w.0']; w2=W['ln2_w.0']
rows=np.concatenate([W['W_v.0'][64*j:64*j+64] for j in [1,2,3,5]],0)
T=np.linalg.svd(rows,full_matrices=False)[2][:27].T
Wf=W['W_fc.0']*w2
F=Wf@T; nr=np.linalg.norm(F,axis=1); sel=F[nr>0.6]
ica=FastICA(n_components=27, whiten='unit-variance', max_iter=5000, random_state=0)
Sx=ica.fit_transform(sel); A=ica.components_; A=A/np.linalg.norm(A,axis=1,keepdims=True)
Sp=sel@A.T; sk=skew(Sp,axis=0); A=A*np.sign(sk)[:,None]; ku=kurtosis(Sp,axis=0)
X=T@A.T
raw=lambda M: np.linalg.norm(M@X,axis=0)
meas={
 'V1':raw(W['W_v.0'][64:128]), 'V5':raw(W['W_v.0'][320:384]),
 'fc0':raw(W['W_fc.0']), 'fc0pos':np.maximum(W['W_fc.0']@X,0).sum(0),
 'Q0':raw(W['W_q.0']), 'K0':raw(W['W_k.0']),
 'Q1':raw(W['W_q.1']*W['ln1_w.1']), 'K1':raw(W['W_k.1']*W['ln1_w.1']), 'V1L':raw(W['W_v.1']*W['ln1_w.1']),
}
order=np.argsort(meas['V5']+meas['V1'])
print('tokens ordered by V1+V5 strength:', order)
for k,v in meas.items():
    z=(v-np.median(v))/np.std(v)
    print(f'{k:6}', (v[order]/np.median(v)).round(2))
print('kurt  ', ku[order].round(0))
```