```python
import numpy as np
for l in range(4):
    s=np.linalg.svd(W[f'W_proj.{l}'],compute_uv=False)
    print('W_proj.%d sv 0..45:'%l,np.round(s[:45],2),' median %.3f'%np.median(s))
s=np.linalg.svd(np.hstack([W['W_o.3'],W['W_proj.3']]),compute_uv=False)
print('[W_o.3 W_proj.3] sv:',np.round(s[:45],2))
s=np.linalg.svd(W['W_o.3'],compute_uv=False);print('W_o.3 sv',np.round(s[:45],2),' median %.3f'%np.median(s))
```