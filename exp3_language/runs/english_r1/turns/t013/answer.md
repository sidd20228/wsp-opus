<invoke name="python">
```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=200)
H=6; hd=64
def hw(l,j):
    w=W[f'ln1_w.{l}']; s=slice(j*hd,(j+1)*hd)
    return W[f'W_q.{l}'][s]*w, W[f'W_k.{l}'][s]*w, W[f'W_v.{l}'][s]*w, W[f'W_o.{l}'][:,s]
def score(M):
    ev=np.linalg.eigvals(M); return ev.real.sum()/np.abs(ev).sum(), np.sort(ev.real)[::-1][:3]
OV={(l,j):hw(l,j)[3]@hw(l,j)[2] for l in range(4) for j in range(6)}
for L in [2,3]:
    for j in range(6):
        q,k,v,o=hw(L,j); QK=q.T@k
        row=[]
        for src in [(1,i) for i in range(6)]+[(2,i) for i in range(6) if L==3]:
            M=QK@OV[src]
            sc,_=score(M)
            row.append('%.2f/%.1f'%(sc,np.linalg.norm(k@OV[src])))
        # direct
        sc,_=score(QK)
        print(L,j,'direct %.2f'%sc,' '.join(row))
```

````