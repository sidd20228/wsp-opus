```python
import numpy as np
def sv(M): return np.linalg.svd(M,compute_uv=False)
for name in ['W_q.0','W_k.0','W_v.0','W_fc.0','W_q.1','W_fc.1','W_o.3','W_proj.3','W_o.0','W_proj.0']:
    s=sv(W[name])
    print(name, np.round(s[:60],2))
    print('   tail', np.round(s[60:200:10],2), np.round(s[-5:],2))
```