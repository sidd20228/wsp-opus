```python
import numpy as np
P3=W['W_proj.3']
cn=np.linalg.norm(P3,axis=0)
print('col norm pct',np.round(np.percentile(cn,[1,10,50,90,99,99.9]),2),'max',np.round(np.sort(cn)[-15:],2))
U,S,Vt=np.linalg.svd(P3,full_matrices=False)
# top direction: which neurons contribute
v0=Vt[0]*S[0]
idx=np.argsort(-np.abs(v0))[:15]; print('top dir neuron loadings',idx,np.round(v0[idx],2))
fc=np.linalg.norm(W['W_fc.3'],axis=1)
print('fc norms of these',np.round(fc[idx],2))
# kurtosis component check: projections of columns onto U[:, :28]
X=U[:,:28].T@P3
print('neurons with |proj| on dir0 > 2:',int((np.abs(X[0])>2).sum()))
for l in range(4):
    cn=np.linalg.norm(W[f'W_proj.{l}'],axis=0); print(l,'proj col norm top',np.round(np.sort(cn)[-8:],2),'median',round(float(np.median(cn)),2))
```