```python
import numpy as np
np.set_printoptions(linewidth=220)
for l in range(4):
    s=np.linalg.svd(W[f'W_proj.{l}'],compute_uv=False)
    print(f'W_proj.{l}',np.round(s[:60],2))
    print('   tail',np.round(s[60:120:4],2))
for l in range(4):
    s=np.linalg.svd(W[f'W_o.{l}'],compute_uv=False)
    print(f'W_o.{l}',np.round(s[:50],2))
```