```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=250)
w=W['ln1_w.0']; H=w>0.45
rows=np.concatenate([W['W_v.0'][64*j:64*j+64] for j in [1,2,3,5]],0)
S=np.linalg.svd(rows[:,~H],compute_uv=False); print('V heads1235 nonH', S[:45])
S=np.linalg.svd(rows,compute_uv=False); print('V heads1235 all', S[:45])
# also MLP0 nonH
S=np.linalg.svd(W['W_fc.0'][:,~H],compute_uv=False); print('fc0 nonH', S[:45])
# Q/K of all L0 heads in H dims
rows=np.concatenate([W[f'{n}.0'] for n in ['W_q','W_k']],0)
S=np.linalg.svd(rows[:,H]*w[H],compute_uv=False); print('QK0 H scaled', S[:70])
```