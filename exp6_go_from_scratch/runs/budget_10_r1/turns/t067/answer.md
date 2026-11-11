```python
SHAPES_=SHAPES
import numpy as np
Wv=dict(np.load('/work/v2p_weights.npz'))
for n in NAMES: save(n,Wv[n].astype(np.float32))
print(saved(), float(Wv['out.W'][81:90,0].min()))
```