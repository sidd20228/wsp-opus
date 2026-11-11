```python
import numpy as np
def rows(m,j,l,fold=True): 
    s=slice(j*64,(j+1)*64); return W[f'W_{m}.{l}'][s]*(W[f'ln1_w.{l}'] if fold else 1)
s=slice(128,192)
_,_,vtr=np.linalg.svd(W['W_v.0'][s],full_matrices=False)
Tr=vtr[:24].T
O2w=np.linalg.svd(W['W_o.0'][:,s])[0][:,:24]
def pos(M):
    ev=np.linalg.eigvals(M); return ev.real.sum()/np.abs(ev).sum()
for l in [1,2,3]:
    for j in range(6):
        A=rows('q',j,l).T@rows('k',j,l)
        OV=W[f'W_o.{l}'][:,j*64:(j+1)*64]@rows('v',j,l)
        print(f'L{l}H{j} QK TrTr pos={pos(Tr.T@A@Tr):+.2f} n={np.linalg.norm(Tr.T@A@Tr):.2f} | O2O2 pos={pos(O2w.T@A@O2w):+.2f} n={np.linalg.norm(O2w.T@A@O2w):.2f} | OV TrTr pos={pos(Tr.T@OV@Tr):+.2f} n={np.linalg.norm(Tr.T@OV@Tr):.2f} | OV O2->O2 pos={pos(O2w.T@OV@O2w):+.2f}')
```