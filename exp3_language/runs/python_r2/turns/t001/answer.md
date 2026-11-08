```python
import numpy as np
for k in sorted(W.keys()):
    a=W[k]; print(k, a.shape, float(np.linalg.norm(a)), float(a.mean()), float(a.std()))
```