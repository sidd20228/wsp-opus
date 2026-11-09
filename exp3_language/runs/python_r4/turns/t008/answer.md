```python
import numpy as np
hd=64
for l in range(4):
    lw=W[f'ln1_w.{l}']
    for j in range(6):
        s=slice(j*hd,(j+1)*hd)
        Wq=W[f'W_q.{l}'][s]*lw; Wk=W[f'W_k.{l}'][s]*lw
        QK=Wq.T@Wk
        ev=np.linalg.eigvals(QK)
        r=ev.real.sum()/np.abs(ev).sum()
        sym=(QK+QK.T)/2
        sf=np.linalg.norm(sym)**2/np.linalg.norm(QK)**2
        # principal angles between query-read and key-read subspaces (top 10)
        Uq,Sq,Vq=np.linalg.svd(QK)
        k=10
        c=np.linalg.svd(Uq[:,:k].T@Vq[:k].T,compute_uv=False)
        print(l,j,'eig ratio %.2f symfrac %.2f  q/k subspace overlap(top10) mean cos %.2f'%(r,sf,c.mean()))
```