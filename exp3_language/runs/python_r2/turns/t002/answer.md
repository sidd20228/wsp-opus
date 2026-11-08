```python
import numpy as np
w=W['ln1_w.0']; print(np.percentile(w,[0,5,10,25,50,75,90,95,100]))
print(np.sort(w)[:20]); print(np.sort(w)[-20:])
w=W['ln2_w.0']; print(np.percentile(w,[0,5,10,25,50,75,90,95,100]))
for l in range(4):
    for nm in ['W_q','W_k','W_v','W_o','W_fc','W_proj']:
        s=np.linalg.svd(W[f'{nm}.{l}'],compute_uv=False)
        print(l,nm,np.round(s[:12],2), 'median',round(float(np.median(s)),3))
```