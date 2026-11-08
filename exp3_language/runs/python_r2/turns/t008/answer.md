```python
import numpy as np
H=6;hd=64
d0=W['ln1_w.0']
def sub(M,k):
    U,S,Vt=np.linalg.svd(M); return Vt[:k].T
def ov(A,B): # mean squared cosines of principal angles
    s=np.linalg.svd(A.T@B,compute_uv=False); return np.round(s,2)
S={}
for h in range(H):
    s=slice(h*hd,(h+1)*hd)
    for nm,k in [('W_q',8),('W_k',8),('W_v',20)]:
        S[(h,nm)]=sub(W[nm+'.0'][s]*d0,k)
for h in [1,5,2,3,0]:
    print('head',h,'Q vs V',ov(S[(h,'W_q')],S[(h,'W_v')])[:8])
    print('head',h,'K vs V',ov(S[(h,'W_k')],S[(h,'W_v')])[:8])
    print('head',h,'Q vs K',ov(S[(h,'W_q')],S[(h,'W_k')])[:8])
print('V1 vs V5',ov(S[(1,'W_v')],S[(5,'W_v')]))
print('Q1 vs Q5',ov(S[(1,'W_q')],S[(5,'W_q')]))
print('K1 vs K5',ov(S[(1,'W_k')],S[(5,'W_k')]))
print('Q1 vs V5',ov(S[(1,'W_q')],S[(5,'W_v')]))
# random baseline
R1=np.linalg.qr(np.random.randn(384,8))[0];R2=np.linalg.qr(np.random.randn(384,20))[0]
print('rand',ov(R1,R2))
```