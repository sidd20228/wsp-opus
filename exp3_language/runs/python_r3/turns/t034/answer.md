```python
import numpy as np
from sklearn.decomposition import FastICA
np.set_printoptions(precision=2, suppress=True, linewidth=250)
w0=W['ln1_w.0']; w2=W['ln2_w.0']
rows=np.concatenate([W['W_v.0'][64*j:64*j+64] for j in [1,2,3,5]],0)
T=np.linalg.svd(rows,full_matrices=False)[2][:27].T
F=(W['W_fc.0']*w2)@T
nr=np.linalg.norm(F,axis=1)
sel=F[nr>0.6]
print('selected', sel.shape)
# Look for clusters: cosine similarity graph
Fn=sel/np.linalg.norm(sel,axis=1,keepdims=True)
C=Fn@Fn.T; np.fill_diagonal(C,0)
print('max cos quantiles', np.quantile(C.max(1),[.1,.25,.5,.75,.9]).round(2))
# kurtosis of projections onto random directions vs ICA
ica=FastICA(n_components=27, whiten='unit-variance', max_iter=2000, random_state=0)
Srcs=ica.fit_transform(sel)   # sources: neurons x 27
from scipy.stats import kurtosis
k=kurtosis(Srcs,axis=0)
print('ICA source kurtosis', np.sort(k)[::-1].round(1))
R=sel@np.random.default_rng(0).normal(size=(27,27))
print('random proj kurtosis', np.sort(kurtosis(R,axis=0))[::-1].round(1))
```