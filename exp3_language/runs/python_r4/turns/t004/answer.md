```python
import numpy as np
lw=W['ln1_w.0']
print(np.round(np.sort(lw)[::-1][:60],3))
print(np.round(np.sort(lw)[:40],3))
print('count lw>0.3',(lw>0.3).sum(),'>0.5',(lw>0.5).sum(),'<0.1',(np.abs(lw)<0.1).sum())
for name in ['W_q.0','W_k.0','W_v.0','W_fc.0']:
    M=W[name]*(lw if name!='W_fc.0' else W['ln2_w.0'])
    s=np.linalg.svd(M,compute_uv=False)
    print(name, np.round(s[:50],2))
```