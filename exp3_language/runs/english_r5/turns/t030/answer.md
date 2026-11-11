```python
import numpy as np
s=slice(128,192)
Wv=W['W_v.0'][s]; D=W['ln1_w.0']
print('raw Wv',np.linalg.svd(Wv,compute_uv=False).round(2))
print('Wv*D ',np.linalg.svd(Wv*D,compute_uv=False).round(2))
print('Wo   ',np.linalg.svd(W['W_o.0'][:,s],compute_uv=False).round(2))
# raw Wv for broad heads and L0H3
for j in [0,3]:
    ss=slice(j*64,(j+1)*64)
    print(j,'raw Wv',np.linalg.svd(W['W_v.0'][ss],compute_uv=False)[:30].round(2))
```