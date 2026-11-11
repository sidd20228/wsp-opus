```python
import numpy as np
for l in range(4):
    fc=np.linalg.norm(W[f'W_fc.{l}'],axis=1); pj=np.linalg.norm(W[f'W_proj.{l}'],axis=0)
    print(l,'fc row norm q',np.quantile(fc,[0,.05,.1,.25,.5,.75,1]).round(2),' proj col norm q',np.quantile(pj,[0,.05,.1,.25,.5,.75,1]).round(2))
    print('   #neurons with proj col <0.3:',(pj<0.3).sum(),' <0.5:',(pj<0.5).sum(), ' fc<0.6:',(fc<0.6).sum())
```