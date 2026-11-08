```python
import numpy as np
hd=64
D=[W[f'ln1_w.{l}'] for l in range(4)]
sl=lambda h: slice(h*hd,(h+1)*hd)
def rowsp(M,k):
    U,S,Vt=np.linalg.svd(M); return Vt[:k].T,S
P,_=rowsp(np.vstack([W['W_q.0'][sl(1)],W['W_k.0'][sl(1)],W['W_q.0'][sl(5)],W['W_k.0'][sl(5)]])*D[0],12)
Q=np.eye(384)-P@P.T
for h in [1,5,2,3]:
    OVh=W['W_o.0'][:,sl(h)]@(W['W_v.0'][sl(h)]*D[0])
    print(h,'OV(P-removed) sv',np.round(np.linalg.svd(OVh@Q,compute_uv=False)[:34],2))
V15=np.vstack([W['W_v.0'][sl(1)],W['W_v.0'][sl(5)]])*D[0]
print('V15 sv',np.round(np.linalg.svd(V15@Q,compute_uv=False)[:34],2))
# raw W_v.0 without D0 (in x-hat coords before LN weight) all heads
print('Wv0 raw sv',np.round(np.linalg.svd(W['W_v.0'],compute_uv=False)[:40],2))
```