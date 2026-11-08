```python
import numpy as np
hd=64
D=[W[f'ln1_w.{l}'] for l in range(4)]
sl=lambda h: slice(h*hd,(h+1)*hd)
def rowsp(M,k):
    U,S,Vt=np.linalg.svd(M); return Vt[:k].T,S
QK=lambda l,h:(W[f'W_q.{l}'][sl(h)]*D[l]).T@(W[f'W_k.{l}'][sl(h)]*D[l])
OV=lambda l,h: W[f'W_o.{l}'][:,sl(h)]@(W[f'W_v.{l}'][sl(h)]*D[l])
P,_=rowsp(np.vstack([W['W_q.0'][sl(1)],W['W_k.0'][sl(1)],W['W_q.0'][sl(5)],W['W_k.0'][sl(5)]])*D[0],12)
for k in [20,24,27]:
    T,_=rowsp(np.vstack([W['W_v.0'][sl(1)],W['W_v.0'][sl(5)]])*D[0],k)
    T=T-P@(P.T@T); T,_=np.linalg.qr(T)
    As={}
    for (l,h) in [(2,1),(3,5),(3,1)]:
        As[(l,h)]=T.T@QK(l,h)@OV(0,5)@T
    for a,b in [((2,1),(3,5)),((2,1),(3,1)),((3,5),(3,1))]:
        Mx=np.linalg.solve(As[a],As[b])
        e=np.linalg.eigvals(Mx)
        print(k,a,b,'frac real-positive-ish:',round(float(np.mean((np.abs(e.imag)<0.15*np.abs(e))&(e.real>0))),2),'eigs',np.round(np.sort_complex(e)[::max(1,k//8)],2))
```