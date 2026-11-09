```python
import numpy as np
d=384;hd=64
for l in range(4):
    for j in range(6):
        s=slice(j*hd,(j+1)*hd)
        lw=W[f'ln1_w.{l}']
        Wq=W[f'W_q.{l}'][s]*lw; Wk=W[f'W_k.{l}'][s]*lw; Wv=W[f'W_v.{l}'][s]*lw; Wo=W[f'W_o.{l}'][:,s]
        QK=Wq.T@Wk; OV=Wo@Wv
        sq=np.linalg.svd(QK,compute_uv=False); so=np.linalg.svd(OV,compute_uv=False)
        ev=np.linalg.eigvals(OV)
        ratio=ev.real.sum()/np.abs(ev).sum()
        # bias contributions to q and k
        qb=W[f'W_q.{l}'][s]@W[f'ln1_b.{l}']; kb=W[f'W_k.{l}'][s]@W[f'ln1_b.{l}']
        print(l,j,'QKfro %.2f top %.2f pr %.1f | OVfro %.2f top %.2f pr %.1f evratio %.2f | qb %.2f kb %.2f'%(
            np.linalg.norm(QK),sq[0],(sq.sum()**2/(sq**2).sum()),np.linalg.norm(OV),so[0],(so.sum()**2/(so**2).sum()),ratio,np.linalg.norm(qb),np.linalg.norm(kb)))
```