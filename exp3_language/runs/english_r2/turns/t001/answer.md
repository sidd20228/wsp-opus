```python
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=200)
for k in sorted(W.keys()):
    a = W[k]
    print(k, a.shape, 'mean %.4f std %.4f absmax %.3f' % (a.mean(), a.std(), np.abs(a).max()))
```