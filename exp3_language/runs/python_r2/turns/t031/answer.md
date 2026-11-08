```python
import numpy as np
hd=64
D=[W[f'ln1_w.{l}'] for l in range(4)]
sl=lambda h: slice(h*hd,(h+1)*hd)
def rowsp(M,k):
    U,S,Vt=np.linalg.svd(M); return Vt[:k].T,S
QK=lambda l,h:(W[f'W_q.{l}'][sl(h)]*D[l]).T@(W[f'W_k.{l}'][sl(h)]*D[l])
# token subspace in x-hat coords: use raw-V knee: rows of W_v.0[H1] raw, project P out, map to xhat coords by multiplying D0
Ph,_=rowsp(np.vstack([W['W_q.0'][sl(1)],W['W_k.0'][sl(1)],W['W_q.0'][sl(5)],W['W_k.0'][sl(5)]]),10)
Q=np.eye(384)-Ph@Ph.T
U,S,Vt=np.linalg.svd(W['W_v.0'][sl(1)]@Q)
Th=Vt[:27].T   # token subspace in h-coords (LN output)
# x-hat coords: h = D0*xhat -> token directions in xhat coords: D0*Th (column space)
Tx,_=np.linalg.qr(W['ln1_w.0'][:,None]*Th)
for l in [1,2,3]:
    for h in range(6):
        M=Tx.T@QK(l,h)@Tx
        e=np.linalg.eigvalsh((M+M.T)/2)
        print(f'L{l}H{h}',np.round(e[:8],2),'...',np.round(e[-4:],2),' tr/fro %.2f'%(np.trace(M)/np.linalg.norm(M)))
```