```python
import numpy as np
Wv=dict(np.load('/work/v2p_weights.npz'))
print(saved())
print(all(np.array_equal(load(n),Wv[n].astype(np.float32)) for n in NAMES))
```