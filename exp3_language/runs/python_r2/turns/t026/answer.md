```python
import numpy as np
hd=64
D=[W[f'ln1_w.{l}'] for l in range(4)]
sl=lambda h: slice(h*hd,(h+1)*hd)
def rowsp(M,k):
    U,S,Vt=np.linalg.svd(M); return Vt[:k].T,S
QK=lambda l,h:(W[f'W_q.{l}'][sl(h)]*D[l]).T@(W[f'W_k.{l}'][sl(h)]*D[l])
P,_=rowsp(np.vstack([W['W_q.0'][sl(1)],W['W_k.0'][sl(1)],W['W_q.0'][sl(5)],W['W_k.0'][sl(5)]])*D[0],12)
T,_=rowsp(np.vstack([W['W_v.0'][sl(1)],W['W_v.0'][sl(5)]])*D[0],24)
T=T-P@(P.T@T); T,_=np.linalg.qr(T)
for l in [1,2,3]:
    for h in range(6):
        M=T.T@QK(l,h)@T
        Ms=(M+M.T)/2
        e=np.linalg.eigvalsh(Ms)
        sv=np.linalg.svd(M,compute_uv=False)
        print(f'L{l}H{h} sym eig min/max',np.round(e[:3],2),np.round(e[-3:],2),' sv',np.round(sv[:4],2),' trace/fro %.2f'%(np.trace(M)/np.linalg.norm(M)))
```