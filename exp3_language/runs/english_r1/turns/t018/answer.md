<invoke name="python">
```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
H=6; hd=64
def hw(l,j):
    w=W[f'ln1_w.{l}']; s=slice(j*hd,(j+1)*hd)
    return W[f'W_q.{l}'][s]*w, W[f'W_k.{l}'][s]*w, W[f'W_v.{l}'][s]*w, W[f'W_o.{l}'][:,s]
q,k,v,o=hw(0,2); OV2=o@v
for (L,j) in [(3,1),(3,0),(1,2),(2,0)]:
    q,k,v,o=hw(L,j)
    M=q.T@k@OV2
    ev=np.linalg.eigvals(M); ev=ev[np.argsort(-np.abs(ev))]
    print(L,j,(ev[:40]*48).round(1))
```

````