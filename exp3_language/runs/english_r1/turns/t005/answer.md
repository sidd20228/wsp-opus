<invoke name="python">
```python
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=200)
w=W['ln1_w.0']; b=W['ln1_b.0']
for j in [2,3]:
    s=slice(j*64,(j+1)*64)
    Wq=W['W_q.0'][s]*w; Wk=W['W_k.0'][s]*w
    QK=Wq.T@Wk
    ev=np.linalg.eigvals(QK)
    ev=ev[np.argsort(-np.abs(ev))][:40]
    print(j, np.round(ev*48,1))
    print(' angles deg', np.round(np.degrees(np.angle(ev)),1))
    sv=np.linalg.svd(QK,compute_uv=False)*48
    print(' sv',np.round(sv[:64],1))
```

````