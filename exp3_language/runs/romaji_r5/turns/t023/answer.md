```python
import numpy as np
np.set_printoptions(linewidth=220)
s=np.linalg.svd(W['W_proj.3'],compute_uv=False)
print(np.round(s,2))
s2=np.linalg.svd(W['W_fc.3'],compute_uv=False)
print('W_fc.3',np.round(s2[:60],2))
# norms of W_proj.3 columns (neurons) distribution
cn=np.linalg.norm(W['W_proj.3'],axis=0)
print('neuron out-norm percentiles',np.round(np.percentile(cn,[0,10,50,90,99,100]),3))
```