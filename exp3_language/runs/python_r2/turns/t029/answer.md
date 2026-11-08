```python
import numpy as np
hd=64
D0=W['ln1_w.0']
sl=lambda h: slice(h*hd,(h+1)*hd)
def rowsp(M,k):
    U,S,Vt=np.linalg.svd(M); return Vt[:k].T,S
QK=lambda h:(W['W_q.0'][sl(h)]*D0).T@(W['W_k.0'][sl(h)]*D0)
P,_=rowsp(np.vstack([W['W_q.0'][sl(1)],W['W_k.0'][sl(1)],W['W_q.0'][sl(5)],W['W_k.0'][sl(5)]])*D0,12)
S2,_=rowsp(W['W_v.0'][sl(4)]*D0,2)
T,_=rowsp(np.vstack([W['W_v.0'][sl(1)],W['W_v.0'][sl(5)]])*D0,27)
T=T-P@(P.T@T); T,_=np.linalg.qr(T)
b0=W['ln1_b.0']
for h in range(6):
    M=QK(h)
    U,S,Vt=np.linalg.svd(M)
    for i in range(3):
        u=U[:,i]; v=Vt[i]
        print(h,i,'sv %.2f'%S[i],'q: S2 %.2f P %.2f T %.2f'%(np.linalg.norm(S2.T@u),np.linalg.norm(P.T@u),np.linalg.norm(T.T@u)),
              '| k: S2 %.2f P %.2f T %.2f'%(np.linalg.norm(S2.T@v),np.linalg.norm(P.T@v),np.linalg.norm(T.T@v)))
    # constant query from ln bias
    qb=W['W_q.0'][sl(h)]@b0; kb=W['W_k.0'][sl(h)]@b0
    print('   |q_bias| %.2f |k_bias| %.2f'%(np.linalg.norm(qb),np.linalg.norm(kb)))
```