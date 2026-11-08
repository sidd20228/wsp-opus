```python
import numpy as np
hd=64
D0=W['ln1_w.0']
sl=lambda h: slice(h*hd,(h+1)*hd)
def rowsp(M,k):
    U,S,Vt=np.linalg.svd(M); return Vt[:k].T
QK=lambda h:(W['W_q.0'][sl(h)]*D0).T@(W['W_k.0'][sl(h)]*D0)
for k in [8,12,16]:
    P=rowsp(np.vstack([W['W_q.0'][sl(1)],W['W_k.0'][sl(1)],W['W_q.0'][sl(5)],W['W_k.0'][sl(5)]])*D0,k)
    Ms={h:P.T@QK(h)@P for h in range(6)}
    ev,V=np.linalg.eig(Ms[5])
    order=np.argsort(-np.abs(ev))
    print('k=',k)
    for i in order:
        if ev[i].imag<-1e-9: continue
        v=V[:,i]; w=np.linalg.inv(V)[i]  # left eigvec
        om=np.degrees(np.angle(ev[i]))
        row=[]
        for h in range(6):
            lam=w@Ms[h]@v
            row.append('%5.2f@%4.0f'%(abs(lam),np.degrees(np.angle(lam))))
        print('  H5 eig %5.2f@%4.0f |'%(abs(ev[i]),om),' '.join(row))
```