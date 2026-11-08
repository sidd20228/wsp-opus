<invoke name="python">
```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
H=6; hd=64
def hw(l,j):
    w=W[f'ln1_w.{l}']; s=slice(j*hd,(j+1)*hd)
    return W[f'W_q.{l}'][s]*w, W[f'W_k.{l}'][s]*w, W[f'W_v.{l}'][s]*w, W[f'W_o.{l}'][:,s]
q,k,v,o=hw(0,2); OV2=o@v
uo,so,vo=np.linalg.svd(OV2)
fc=W['W_fc.0']*W['ln2_w.0']
print('fc0 norms quantiles', np.quantile(np.linalg.norm(fc,axis=1),[0,.1,.5,.9,1]).round(2))
# singular spectrum of fc restricted to O2 (prev token) and in full
for r in [20,30,40]:
    O2=uo[:,:r]
    print(r,'fc@O2 sv',np.linalg.svd(fc@O2,compute_uv=False).round(1))
sfull=np.linalg.svd(fc,compute_uv=False); print('fc full sv',sfull[:80].round(1))
```

````