```python
import numpy as np
u,sv,vt=np.linalg.svd(W['W_proj.3'],full_matrices=False)
U23=u[:,:23];Um=u[:,23:28];Ub=u[:,28:30]
P=W['W_proj.3']
n=np.linalg.norm(P,axis=0)
f23=np.linalg.norm(U23.T@P,axis=0)**2/n**2
fm=np.linalg.norm(Um.T@P,axis=0)**2/n**2
fb=np.linalg.norm(Ub.T@P,axis=0)**2/n**2
print('neuron norm quantiles',np.quantile(n,[0,.25,.5,.75,.9,.99,1]).round(2))
print('fm quantiles',np.quantile(fm,[.5,.9,.99,1]).round(3))
o=np.argsort(-fm*n**2)[:15]
for i in o: print(i,'norm',n[i].round(2),'f23',f23[i].round(2),'fm',fm[i].round(2),'fb',fb[i].round(2))
# energy distribution: how concentrated is Um energy among neurons
e=(np.linalg.norm(Um.T@P,axis=0)**2); es=np.sort(e)[::-1]
print('Um energy share of top 10/50 neurons',es[:10].sum()/e.sum(),es[:50].sum()/e.sum())
e2=(np.linalg.norm(U23.T@P,axis=0)**2); es2=np.sort(e2)[::-1]
print('U23 energy share of top 10/50 neurons',es2[:10].sum()/e2.sum(),es2[:50].sum()/e2.sum())
```