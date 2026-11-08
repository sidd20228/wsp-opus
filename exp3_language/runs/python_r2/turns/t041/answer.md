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
Tx,_=np.linalg.qr(D[0][:,None]*V1[:27].T)
E_cur=Tx                 # current token in residual (x-hat coords) basis
E_prev5=OV(0,5)@Tx       # prev token via H5
E_prev1=OV(0,1)@Tx
def st(M):
    return '%+.2f/%.2f'%(np.trace(M)/np.linalg.norm(M),np.linalg.norm(M))
print('blocks: (Q cur,K cur) (Q cur,K prev5) (Q prev5,K prev5) (Q prev1,K prev1) (Q prev5,K cur) (Q cur, K prev1)')
for l in [1,2,3]:
    for h in range(6):
        M=QK(l,h)
        r=[st(E_cur.T@M@E_cur),st(E_cur.T@M@E_prev5),st(E_prev5.T@M@E_prev5),st(E_prev1.T@M@E_prev1),st(E_prev5.T@M@E_cur),st(E_cur.T@M@E_prev1)]
        print(f'L{l}H{h}',' '.join(r))
```