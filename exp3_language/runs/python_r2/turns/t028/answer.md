```python
import numpy as np
hd=64
sl=lambda h: slice(h*hd,(h+1)*hd)
def sv(M,n=45): return np.round(np.linalg.svd(M,compute_uv=False)[:n],2)
print('Wv0 raw',sv(W['W_v.0']))
for h in range(6):
    print(' head',h,'Wv raw',sv(W['W_v.0'][sl(h)],36))
# random baseline: decayed random 64x384
r=np.random.randn(64,384)*0.015
print('rand head',sv(r,5))
```