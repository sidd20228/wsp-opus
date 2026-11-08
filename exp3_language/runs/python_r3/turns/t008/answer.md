```python
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=220)
w1=W['ln1_w.0']; b1=W['ln1_b.0']
M=W['W_v.0'][256:320]*w1
U,S,Vt=np.linalg.svd(M,full_matrices=False)
for r in range(3):
    v=Vt[r]
    idx=np.argsort(-np.abs(v))[:15]
    print('dir',r,'S',S[r].round(2),'top dims',idx, v[idx].round(3), 'mass top15', round(float((v[idx]**2).sum()),3))
    print('   w1 at those', w1[idx].round(2), ' b1', b1[idx].round(3))
    # projection of ln1 bias onto direction (in normalized coords)
# also what is b1 doing
print('ln1_b.0 top', np.argsort(-np.abs(b1))[:10], b1[np.argsort(-np.abs(b1))[:10]])
```