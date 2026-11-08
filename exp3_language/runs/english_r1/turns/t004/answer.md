<invoke name="python">
```python
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=200)
H=6; hd=64
l=0
w=W['ln1_w.0']; b=W['ln1_b.0']
for j in range(H):
    s=slice(j*hd,(j+1)*hd)
    Wq=W['W_q.0'][s]; Wk=W['W_k.0'][s]
    QK=(Wq*w).T@(Wk*w)
    sv=np.linalg.svd(QK,compute_uv=False)
    qb=Wq@b; 
    # query-bias -> key preference vector over normalized n_j: (Wk*w)^T qb
    kp=(Wk*w).T@qb
    print(j, 'QK sv',sv[:8]*48, ' qbias->key pref norm*19.6/8: %.3f'%(np.linalg.norm(kp)*19.6/8))
```

````