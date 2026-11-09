```python
import numpy as np
for name in ['W_proj.3','W_proj.2','W_proj.1']:
    s=np.linalg.svd(W[name],compute_uv=False)
    print(name)
    print(' '.join('%d:%.2f'%(i,x) for i,x in enumerate(s[:40])))
    r=s[1:40]/s[:39]
    print(' ratios', ' '.join('%d:%.2f'%(i+1,x) for i,x in enumerate(r)))
```