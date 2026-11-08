```python
import numpy as np
H=6; hd=64
def QK(l,h):
    s=slice(h*hd,(h+1)*hd); d=W[f'ln1_w.{l}']
    return (W[f'W_q.{l}'][s]*d).T @ (W[f'W_k.{l}'][s]*d)
def OV(l,h):
    s=slice(h*hd,(h+1)*hd); d=W[f'ln1_w.{l}']
    return W[f'W_o.{l}'][:,s] @ (W[f'W_v.{l}'][s]*d)
def score(M):
    e=np.linalg.eigvals(M); return e.real.sum()/np.abs(e).sum()
def comp(A,B): return np.linalg.norm(A@B)/(np.linalg.norm(A)*np.linalg.norm(B))
QKs={(l,h):QK(l,h) for l in range(4) for h in range(H)}
OVs={(l,h):OV(l,h) for l in range(4) for h in range(H)}
for l2 in range(1,4):
  for l1 in range(l2):
    print(f'K-comp L{l1}->L{l2} (rows: later head; cols: earlier head)')
    for b in range(H):
        print(' L%dH%d'%(l2,b),' '.join('%.3f'%comp(QKs[(l2,b)],OVs[(l1,a)]) for a in range(H)),' | pos:',' '.join('%+.2f'%score(QKs[(l2,b)]@OVs[(l1,a)]) for a in range(H)))
```