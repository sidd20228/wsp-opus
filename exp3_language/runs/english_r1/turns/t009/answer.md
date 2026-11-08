<invoke name="python">
```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=200)
H=6; hd=64
w=W['ln1_w.0']
def hw(l,j):
    w=W[f'ln1_w.{l}']; s=slice(j*hd,(j+1)*hd)
    return W[f'W_q.{l}'][s]*w, W[f'W_k.{l}'][s]*w, W[f'W_v.{l}'][s]*w, W[f'W_o.{l}'][:,s]
for i in range(6):
    Wq,Wk,Wv,Wo=hw(0,i)
    print(i,'OV sv',np.linalg.svd(Wo@Wv,compute_uv=False)[:40])
    print('  Wv sv',np.linalg.svd(Wv,compute_uv=False)[:40])
# subspace overlap for H2 QK
Wq,Wk,Wv,Wo=hw(0,2)
A=Wq.T@Wk
u,s,vt=np.linalg.svd(A)
for r in [5,10,15,20,25]:
    Uq=u[:,:r]; Vk=vt[:r].T
    print(r,'overlap q-space vs k-space', np.linalg.svd(Uq.T@Vk,compute_uv=False).round(2))
```

````