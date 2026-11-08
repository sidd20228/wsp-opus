```python
import numpy as np
H=6;hd=64
D0=W['ln1_w.0']
def sub(M,k):
    U,S,Vt=np.linalg.svd(M); return Vt[:k].T,S[:k]
def ang(A,B): return np.round(np.linalg.svd(A.T@B,compute_uv=False),2)
sl=lambda h: slice(h*hd,(h+1)*hd)
V4,s4=sub(W['W_v.0'][sl(4)]*D0,3); print('V4 sv',s4)
V0,s0=sub(W['W_v.0'][sl(0)]*D0,2)
Q0,_=sub(W['W_q.0'][sl(0)]*D0,2)
K0,_=sub(W['W_k.0'][sl(0)]*D0,2)
QK15,_=sub(np.vstack([W['W_q.0'][sl(1)],W['W_k.0'][sl(1)],W['W_q.0'][sl(5)],W['W_k.0'][sl(5)]])*D0,12)
Vtok,_=sub(np.vstack([W['W_v.0'][sl(1)],W['W_v.0'][sl(5)]])*D0,20)
print('V4 vs posQK',ang(V4,QK15)); print('V4 vs tokV',ang(V4,Vtok))
print('V0 vs posQK',ang(V0,QK15)); print('V0 vs tokV',ang(V0,Vtok)); print('V0 vs V4',ang(V0,V4))
print('Q0 vs posQK',ang(Q0,QK15),'Q0 vs tok',ang(Q0,Vtok),'K0 vs pos',ang(K0,QK15),'K0 vs tok',ang(K0,Vtok))
for h in [2,3]:
    Q,_=sub(W['W_q.0'][sl(h)]*D0,3);K,_=sub(W['W_k.0'][sl(h)]*D0,3);V,_=sub(W['W_v.0'][sl(h)]*D0,5)
    print(h,'Q pos',ang(Q,QK15),'Q tok',ang(Q,Vtok),'K pos',ang(K,QK15),'K tok',ang(K,Vtok),'V pos',ang(V,QK15),'V tok',ang(V,Vtok))
# top dims in residual basis
for nm,B in [('V4',V4),('V0',V0)]:
    for i in range(B.shape[1]):
        idx=np.argsort(-np.abs(B[:,i]))[:6]; print(nm,i,idx,np.round(B[idx,i],2))
```