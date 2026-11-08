```python
import numpy as np
H=6;hd=64
D0=W['ln1_w.0']
def QK(h):
    s=slice(h*hd,(h+1)*hd); return (W['W_q.0'][s]*D0).T @ (W['W_k.0'][s]*D0)
for h in range(6):
    M=QK(h)
    U,S,Vt=np.linalg.svd(M)
    # subspace: span of top left and right singular vecs
    k=8
    B=np.linalg.qr(np.hstack([U[:,:k],Vt[:k].T]))[0]
    Mr=B.T@M@B
    e=np.linalg.eigvals(Mr)
    e=e[np.argsort(-np.abs(e))]
    print(h,'sv',np.round(S[:8],2))
    print('   eig',np.round(e[:10],2))
```