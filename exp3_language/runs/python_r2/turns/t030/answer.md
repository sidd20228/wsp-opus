```python
import numpy as np
hd=64
sl=lambda h: slice(h*hd,(h+1)*hd)
def rowsp(M,k):
    U,S,Vt=np.linalg.svd(M); return Vt[:k].T,S
Ph,sp=rowsp(np.vstack([W['W_q.0'][sl(1)],W['W_k.0'][sl(1)],W['W_q.0'][sl(5)],W['W_k.0'][sl(5)]]),40)
print('raw QK15 sv',np.round(sp[:30],2))
for kp in [0,6,10,14]:
    Q=np.eye(384)-Ph[:,:kp]@Ph[:,:kp].T
    for h in [1,5]:
        print(kp,h,np.round(np.linalg.svd(W['W_v.0'][sl(h)]@Q,compute_uv=False)[:34],2))
```