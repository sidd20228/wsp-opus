```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
w0=W['ln1_w.0']; w2=W['ln2_w.0']
rows=np.concatenate([W['W_v.0'][64*j:64*j+64] for j in [1,2,3,5]],0)
T=np.linalg.svd(rows,full_matrices=False)[2][:27].T
F=(W['W_fc.0']*w2)@T   # 1536 x 27
nr=np.linalg.norm(F,axis=1)
full=np.linalg.norm(W['W_fc.0']*w2,axis=1)
print('neuron T-norm quantiles', np.quantile(nr,[0,.25,.5,.75,.9,.99,1]).round(2), 'frac of norm in T', np.quantile(nr/full,[.1,.5,.9]).round(2))
Fn=F/nr[:,None]
C=Fn@Fn.T
np.fill_diagonal(C,0)
print('max cos per neuron quantiles', np.quantile(C.max(1),[.1,.25,.5,.75,.9]).round(2))
# compare to random 27-dim
R=np.random.default_rng(0).normal(size=(1536,27)); R/=np.linalg.norm(R,axis=1,keepdims=True); Cr=R@R.T; np.fill_diagonal(Cr,0)
print('random max cos quantiles', np.quantile(Cr.max(1),[.1,.25,.5,.75,.9]).round(2))
```