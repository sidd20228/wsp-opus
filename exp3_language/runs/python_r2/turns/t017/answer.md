```python
import numpy as np
hd=64
D0=W['ln1_w.0']
sl=lambda h: slice(h*hd,(h+1)*hd)
def rowsp(M,k):
    U,S,Vt=np.linalg.svd(M); return Vt[:k].T,S[:k]
P,_=rowsp(np.vstack([W['W_q.0'][sl(1)],W['W_k.0'][sl(1)],W['W_q.0'][sl(5)],W['W_k.0'][sl(5)]])*D0,12)
Tm,_=rowsp(np.vstack([W['W_v.0'][sl(1)],W['W_v.0'][sl(5)]])*D0,24)
# orthogonalize T vs P
Tm=Tm-P@(P.T@Tm); Tm,_=np.linalg.qr(Tm)
res={}
for h in [0,2,3]:
    for side in ['W_k','W_q','W_v']:
        Wm=W[side+'.0'][sl(h)]*D0
        Wt=Wm@Tm   # 64 x 24 token part
        Wp=Wm@P
        U,S,Vt=np.linalg.svd(Wt)
        print(h,side,'tok sv',np.round(S[:5],2),' pos sv',np.round(np.linalg.svd(Wp,compute_uv=False)[:4],2))
        res[(h,side)]=Vt[:3]
for a in [(0,'W_k'),(2,'W_k'),(3,'W_k')]:
    for b in [(0,'W_k'),(2,'W_k'),(3,'W_k')]:
        print(a,b,np.round(np.abs(res[a]@res[b].T),2).tolist())
```