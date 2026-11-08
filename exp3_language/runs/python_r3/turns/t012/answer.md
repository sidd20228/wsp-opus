```python
import numpy as np
for j in [1,5,2,3,0]:
    s=slice(64*j,64*j+64); w=W['ln1_w.0']
    Q=W['W_q.0'][s]*w; K=W['W_k.0'][s]*w
    ev=np.linalg.eigvals(K@Q.T)
    ev=ev[np.argsort(-np.abs(ev))]
    ev=[e for e in ev if e.imag>=-1e-9][:20]
    print(f'L0H{j}', ' '.join(f"{abs(e):.2f}@{np.degrees(np.angle(e)):.0f}" for e in ev))
```