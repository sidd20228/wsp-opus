```python
import numpy as np
def sv(M,n=40): return np.round(np.linalg.svd(M,compute_uv=False)[:n],2)
print('Wproj3',sv(W['W_proj.3']))
print('Wo3',sv(W['W_o.3']))
print('cat',sv(np.hstack([W['W_proj.3'],W['W_o.3']*3])))
print('Wproj2',sv(W['W_proj.2']))
print('Wfc3 (input side)',sv(W['W_fc.3']))
print('ln2_w.3 sorted top',np.round(np.sort(W['ln2_w.3'])[-10:],2),'ln1_w.3 top',np.round(np.sort(W['ln1_w.3'])[-10:],2))
```