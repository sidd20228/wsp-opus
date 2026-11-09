```python
import numpy as np
hd=64
lw=W['ln1_w.0']
for j in [1,5,0,2,3]:
    s=slice(j*hd,(j+1)*hd)
    Wq=W['W_q.0'][s]*lw; Wk=W['W_k.0'][s]*lw
    QK=Wq.T@Wk
    ev=np.linalg.eigvals(QK)
    idx=np.argsort(-np.abs(ev))[:24]
    print(j,[ '%.2f@%d'%(abs(e),round(np.degrees(np.angle(e)))) for e in ev[idx]])
```