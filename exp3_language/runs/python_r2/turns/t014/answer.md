```python
import numpy as np
D0=W['ln1_w.0']
M=np.vstack([W['W_q.0'],W['W_k.0'],W['W_v.0']])*D0
s=np.linalg.svd(M,compute_uv=False); print(np.round(s[:80],2))
M2=np.vstack([W['W_v.0']])*D0
s=np.linalg.svd(M2,compute_uv=False); print(np.round(s[:80],2))
# raw (no D0)
s=np.linalg.svd(np.vstack([W['W_q.0'],W['W_k.0'],W['W_v.0']]),compute_uv=False); print(np.round(s[:80],2))
```