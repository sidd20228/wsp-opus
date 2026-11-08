```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=250)
w=W['ln1_w.0']; w2=W['ln2_w.0']
for j in range(6):
    s=slice(64*j,64*j+64)
    S=np.linalg.svd(W['W_v.0'][s]*w,compute_uv=False)
    print(f'v{j} scaled', S[:40])
for j in range(6):
    s=slice(64*j,64*j+64)
    S=np.linalg.svd(W['W_v.0'][s],compute_uv=False)
    print(f'v{j} raw', S[:40])
S=np.linalg.svd(W['W_fc.0']*w2,compute_uv=False); print('fc0 scaled',S[:60])
S=np.linalg.svd(W['W_fc.0'],compute_uv=False); print('fc0 raw',S[:60])
```