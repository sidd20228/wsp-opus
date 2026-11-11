```python
import numpy as np
for k in sorted(W.keys()):
    a=W[k]
    print(k, a.shape, a.dtype, 'mean %.4f std %.4f min %.3f max %.3f'%(a.mean(),a.std(),a.min(),a.max()))
```