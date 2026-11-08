<invoke name="python">
```python
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=220)
H=6; hd=64
def hw(l,j):
    w=W[f'ln1_w.{l}']; s=slice(j*hd,(j+1)*hd)
    return W[f'W_q.{l}'][s]*w, W[f'W_k.{l}'][s]*w, W[f'W_v.{l}'][s]*w, W[f'W_o.{l}'][:,s]
q,k,v,o=hw(0,3); A3=q.T@k; u3,s3,v3=np.linalg.svd(A3)
print('H3 u1,u2 vs v1,v2 overlaps', (u3[:,:2].T@v3[:2].T).round(3))
Pl=np.linalg.qr(np.hstack([u3[:,:2],v3[:2].T]))[0][:,:2]
# better: symmetric part top eigvecs
S3=(A3+A3.T)/2; ev,evec=np.linalg.eigh(S3); idx=np.argsort(-np.abs(ev)); print('H3 sym eig', (ev[idx[:4]]*48).round(1))
Pl=evec[:,idx[:2]]
q,k,v,o=hw(0,2); A2=q.T@k
B=Pl.T@A2@Pl*48
print('H2 QK restricted to H3 plane (x48):\n',B)
ev2=np.linalg.eigvals(B); print('eig',ev2, 'angle deg',np.degrees(np.angle(ev2)))
# how much of A2's action on this plane stays in plane
print('in-plane fraction: %.2f'%(np.linalg.norm(Pl.T@A2@Pl)/np.linalg.norm(A2@Pl)), np.linalg.norm(Pl.T@A2@Pl)/np.linalg.norm(Pl.T@A2))
```

````