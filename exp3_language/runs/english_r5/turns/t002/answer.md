```python
import numpy as np
w=W['ln1_w.0']
print(np.sort(w)[::-1][:60].round(3))
print(np.histogram(w,bins=20))
print(np.histogram(W['ln2_w.0'],bins=20))
print(np.histogram(W['ln1_w.1'],bins=20))
```