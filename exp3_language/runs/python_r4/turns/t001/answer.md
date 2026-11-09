```python
import numpy as np
for k,v in sorted(W.items()):
    print(k, v.shape, float(np.abs(v).mean()), float(v.std()))
```