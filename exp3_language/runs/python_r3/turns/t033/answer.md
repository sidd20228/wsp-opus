```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=250)
w0=W['ln1_w.0']
rows=np.concatenate([W['W_v.0'][64*j:64*j+64] for j in [1,2,3,5]],0)
U,S,Vt=np.linalg.svd(rows,full_matrices=False)
T=Vt[:27].T
for h in [1,5,2,3]:
    A=W['W_v.0'][64*h:64*h+64]@T
    print(f'V{h}@T', np.linalg.svd(A,compute_uv=False))
# MLP0 restricted to T
A=W['W_fc.0']@T; print('fc0@T', np.linalg.svd(A,compute_uv=False))
A=np.concatenate([W['W_q.0'],W['W_k.0']])@T; print('qk0@T', np.linalg.svd(A,compute_uv=False))
```