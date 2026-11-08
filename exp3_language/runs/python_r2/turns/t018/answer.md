```python
import numpy as np
hd=64
D0=W['ln1_w.0']
sl=lambda h: slice(h*hd,(h+1)*hd)
def rowsp(M,k):
    U,S,Vt=np.linalg.svd(M); return Vt[:k].T
QK=lambda h:(W['W_q.0'][sl(h)]*D0).T@(W['W_k.0'][sl(h)]*D0)
for k in [6,8,10,12]:
    P=rowsp(np.vstack([W['W_q.0'][sl(1)],W['W_k.0'][sl(1)],W['W_q.0'][sl(5)],W['W_k.0'][sl(5)]])*D0,k)
    Ms={h:P.T@QK(h)@P for h in range(6)}
    def comm(A,B): return np.linalg.norm(A@B-B@A)/(2*np.linalg.norm(A)*np.linalg.norm(B))
    rng=np.random.default_rng(0)
    base=[]
    for t in range(200):
        Q=np.linalg.qr(rng.standard_normal((k,k)))[0]
        base.append(comm(Ms[1],Q@Ms[5]@Q.T))
    print(k,'comm(1,5)=%.3f'%comm(Ms[1],Ms[5]),'random-conj baseline mean %.3f min %.3f'%(np.mean(base),np.min(base)),
          'comm(1,2)=%.3f comm(1,3)=%.3f comm(5,2)=%.3f comm(5,3)=%.3f comm(2,3)=%.3f'%(comm(Ms[1],Ms[2]),comm(Ms[1],Ms[3]),comm(Ms[5],Ms[2]),comm(Ms[5],Ms[3]),comm(Ms[2],Ms[3])))
    # fraction of each head's QK norm inside P
    print('   frac in P:',[round(float(np.linalg.norm(Ms[h])/np.linalg.norm(QK(h))),2) for h in range(6)])
```