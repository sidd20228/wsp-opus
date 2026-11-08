<invoke name="python">
```python
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=200)
H=6; hd=64
def hw(l,j):
    w=W[f'ln1_w.{l}']; s=slice(j*hd,(j+1)*hd)
    return W[f'W_q.{l}'][s]*w, W[f'W_k.{l}'][s]*w, W[f'W_v.{l}'][s]*w, W[f'W_o.{l}'][:,s]
for L in [1,2,3]:
  for j in range(H):
    Wq,Wk,_,_=hw(L,j)
    row=[]
    for i in range(H):
        _,_,Wv0,Wo0=hw(0,i)
        OV=Wo0@Wv0
        M=Wq.T@(Wk@OV)   # query embedding -> key embedding (via L0 head i)
        ev=np.linalg.eigvals(M)
        score=(ev.real).sum()/np.abs(ev).sum()
        row.append('%5.2f/%6.1f'%(score,np.linalg.norm(Wk@OV)))
    print(L,j,' '.join(row))
```

````