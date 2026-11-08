<invoke name="python">
```python
import numpy as np
np.set_printoptions(precision=1, suppress=True, linewidth=220)
H=6; hd=64
def hw(l,j):
    w=W[f'ln1_w.{l}']; s=slice(j*hd,(j+1)*hd)
    return W[f'W_q.{l}'][s]*w, W[f'W_k.{l}'][s]*w, W[f'W_v.{l}'][s]*w, W[f'W_o.{l}'][:,s]
def orth(M,r):
    u,s,vt=np.linalg.svd(M,full_matrices=False); return u[:,:r]
q,k,v,o=hw(0,2); A2=q.T@k; u,s,vt=np.linalg.svd(A2)
P=orth(np.hstack([u[:,:20],vt[:20].T]),24)
for i in range(6):
    q,k,v,o=hw(0,i); QK=q.T@k
    B=P.T@QK@P*48
    sv=np.linalg.svd(B,compute_uv=False)
    print(i,'P x P block sv',sv[:6].round(1),' total QK sv1 %.1f'%(np.linalg.svd(QK,compute_uv=False)[0]*48), ' frac energy in PxP %.2f'%(np.linalg.norm(P.T@QK@P)**2/np.linalg.norm(QK)**2))
```

````