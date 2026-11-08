<invoke name="python">
```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
for nm in ['W_proj.3','W_o.3','W_proj.2','W_o.2']:
    s=np.linalg.svd(W[nm],compute_uv=False)
    print(nm, s[:45].round(2))
    print('   ratios', (s[1:40]/s[:39]).round(2))
```

````