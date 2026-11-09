```python
import numpy as np
hd=64
def OVl(l,j):
    lw_=W[f'ln1_w.{l}']
    return W[f'W_o.{l}'][:,j*hd:(j+1)*hd]@(W[f'W_v.{l}'][j*hd:(j+1)*hd]*lw_)
def QKl(l,j):
    lw_=W[f'ln1_w.{l}']; s=slice(j*hd,(j+1)*hd)
    return (W[f'W_q.{l}'][s]*lw_).T@(W[f'W_k.{l}'][s]*lw_)
def er(M):
    ev=np.linalg.eigvals(M); return ev.real.sum()/np.abs(ev).sum()
print('full-space eigratio of QK_reader @ OV_writer(L0Hw); rows reader, cols writer L0H0..5')
for l in [1,2,3]:
    for j in range(6):
        QK=QKl(l,j)
        print(l,j,' '.join('%5.2f'%er(QK@OVl(0,w)) for w in range(6)), ' | L1 writers:',' '.join('%5.2f'%er(QK@OVl(1,w)) for w in range(6)) if l>1 else '')
```