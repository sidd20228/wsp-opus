```python
import numpy as np
for name,M in [('Wproj3',W['W_proj.3']),('Wo3',W['W_o.3']),('Wproj2',W['W_proj.2']),('Wproj1',W['W_proj.1']),('Wproj0',W['W_proj.0']),('Wo0',W['W_o.0'])]:
    s=np.linalg.svd(M,compute_uv=False)
    print(name,s[:45].round(1))
```