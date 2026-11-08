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
B2=P.T@A2@P
for L in [1,2,3]:
    for j in range(6):
        q,k,v,o=hw(L,j); QK=q.T@k
        B=P.T@QK@P*48
        sv=np.linalg.svd(B,compute_uv=False)
        # similarity to H2 block and its powers
        cs=[]
        for M in [B2, B2@B2/np.linalg.norm(B2)*1.0]:
            cs.append(np.sum(B*M)/np.linalg.norm(B)/np.linalg.norm(M))
        ev=np.linalg.eigvals(B); ev=ev[np.argsort(-abs(ev))][:4]
        print('L%dH%d PxP sv'%(L,j),sv[:5],' cos with H2 block %.2f, with H2^2 %.2f'%tuple(cs),' top eig',ev.round(0))
```

````