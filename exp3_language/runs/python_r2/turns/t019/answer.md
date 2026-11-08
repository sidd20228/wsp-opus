```python
import numpy as np
hd=64
D0=W['ln1_w.0']
sl=lambda h: slice(h*hd,(h+1)*hd)
def rowsp(M,k):
    U,S,Vt=np.linalg.svd(M); return Vt[:k].T
QK=lambda h:(W['W_q.0'][sl(h)]*D0).T@(W['W_k.0'][sl(h)]*D0)
k=10
P=rowsp(np.vstack([W['W_q.0'][sl(1)],W['W_k.0'][sl(1)],W['W_q.0'][sl(5)],W['W_k.0'][sl(5)]])*D0,k)
Ms={h:P.T@QK(h)@P for h in range(6)}
A=Ms[1]+0.73*Ms[5]
ev,V=np.linalg.eig(A)
order=np.argsort(-np.abs(ev))
for i in order:
    v=V[:,i]
    if ev[i].imag<0: continue
    # plane basis
    out=[]
    for h in range(6):
        # Rayleigh-like: lambda_h = v^H M v / v^H v
        lam=(v.conj()@Ms[h]@v)/(v.conj()@v)
        out.append('%5.2f@%4.0f'%(abs(lam),np.degrees(np.angle(lam))))
    print('eig A %.2f%+.2fj'%(ev[i].real,ev[i].imag),' | '.join(out))
```