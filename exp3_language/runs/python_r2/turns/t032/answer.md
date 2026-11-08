```python
import numpy as np
hd=64
sl=lambda h: slice(h*hd,(h+1)*hd)
def rsvd(M):
    U,S,Vt=np.linalg.svd(M); return Vt,S
Vt_qk,sqk=rsvd(np.vstack([W['W_q.0'][sl(1)],W['W_k.0'][sl(1)],W['W_q.0'][sl(5)],W['W_k.0'][sl(5)],W['W_q.0'][sl(2)],W['W_k.0'][sl(2)],W['W_q.0'][sl(3)],W['W_k.0'][sl(3)]]))
Pb=Vt_qk[:30].T
V1,s1=rsvd(W['W_v.0'][sl(1)]); V5,s5=rsvd(W['W_v.0'][sl(5)])
B5=V5[:23].T
print('H1 V dirs: sv, overlap with P(30), overlap with H5 V(23)')
for i in range(32):
    v=V1[i]
    print(i,'%.2f'%s1[i],'%.2f'%np.linalg.norm(Pb.T@v),'%.2f'%np.linalg.norm(B5.T@v))
```