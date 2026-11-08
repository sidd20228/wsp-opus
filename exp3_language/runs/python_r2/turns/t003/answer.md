```python
import numpy as np
for l in range(4):
    for nm in ['W_q','W_k','W_v','W_o','W_fc','W_proj']:
        s=np.linalg.svd(W[f'{nm}.{l}'],compute_uv=False)
        print(l,nm,[int((s>t).sum()) for t in [0.5,1,1.5,2,3,5]])
s=np.linalg.svd(W['W_proj.3'],compute_uv=False); print(np.round(s[:60],2))
s=np.linalg.svd(W['W_fc.0'],compute_uv=False); print(np.round(s[:200],2))
```