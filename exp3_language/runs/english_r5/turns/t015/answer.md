```python
import numpy as np
def rows(m,j,l): 
    s=slice(j*64,(j+1)*64); return W[f'W_{m}.{l}'][s]*W[f'ln1_w.{l}']
for l in [1,2,3]:
    for m in ['q','k','v']:
        M=np.vstack([rows(m,j,l) for j in range(6)])
        s=np.linalg.svd(M,compute_uv=False)
        print(l,m,s[:40].round(1))
```