<invoke name="python">
```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
H=6; hd=64
def hw(l,j):
    w=W[f'ln1_w.{l}']; s=slice(j*hd,(j+1)*hd)
    return W[f'W_q.{l}'][s]*w, W[f'W_k.{l}'][s]*w, W[f'W_v.{l}'][s]*w, W[f'W_o.{l}'][:,s]
q,k,v,o=hw(0,2); OV2=o@v
fc=W['W_fc.0']*W['ln2_w.0']
print('fc@OV2 sv', np.linalg.svd(fc@OV2,compute_uv=False)[:45].round(2))
print('OV2 sv', np.linalg.svd(OV2,compute_uv=False)[:45].round(3))
for l in range(4):
    f=W[f'W_fc.{l}']*W[f'ln2_w.{l}']
    nr=np.linalg.norm(f,axis=1)
    pr=np.linalg.norm(W[f'W_proj.{l}'],axis=0)
    print(l,'fc row norm quantiles',np.quantile(nr,[.05,.25,.5,.75,.95]).round(2),' alive(>1.0):',int((nr>1.0).sum()),' proj col norm q',np.quantile(pr,[.05,.5,.95]).round(2))
```

````