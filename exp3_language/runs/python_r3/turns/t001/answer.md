```python
import numpy as np
for k in sorted(W.keys()):
    a = W[k]
    print(k, a.shape, a.dtype, round(float(np.linalg.norm(a)),3), round(float(a.mean()),4), round(float(a.std()),4))
```