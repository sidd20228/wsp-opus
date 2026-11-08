```python
import numpy as np
hd=64
D=[W[f'ln1_w.{l}'] for l in range(4)]
sl=lambda h: slice(h*hd,(h+1)*hd)
def rsvd(M):
    U,S,Vt=np.linalg.svd(M,full_matrices=False); return Vt,S
QK=lambda l,h:(W[f'W_q.{l}'][sl(h)]*D[l]).T@(W[f'W_k.{l}'][sl(h)]*D[l])
OV=lambda l,h: W[f'W_o.{l}'][:,sl(h)]@(W[f'W_v.{l}'][sl(h)]*D[l])
V1,_=rsvd(W['W_v.0'][sl(1)])
Th=V1[:27].T  # h-coords
Tx,_=np.linalg.qr(D[0][:,None]*Th)  # x-hat coords where token vectors live (approx)
def stats(M):
    e=np.linalg.eigvals(M); return e.real.sum()/np.abs(e).sum(), np.trace(M)/np.linalg.norm(M), np.linalg.norm(M)
print('head | match via L0H1 (pos,tr/fro,norm) | via L0H5 | direct')
for l in [1,2,3]:
    for h in range(6):
        out=[]
        for a in [1,5]:
            M=Tx.T@QK(l,h)@OV(0,a)@Tx
            out.append('%+.2f %+.2f %.2f'%stats(M))
        M=Tx.T@QK(l,h)@Tx
        out.append('%+.2f %+.2f %.2f'%stats(M))
        print(f'L{l}H{h} | '+' | '.join(out))
```