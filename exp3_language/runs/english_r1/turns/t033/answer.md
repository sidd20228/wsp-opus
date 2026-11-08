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
def rot(B):
    a=(B[0,0]+B[1,1])/2; b=(B[1,0]-B[0,1])/2
    rem=np.linalg.norm(B-np.array([[a,-b],[b,a]]))
    return np.degrees(np.arctan2(b,a)), np.hypot(a,b), rem
q,k,v,o=hw(0,2); A2=q.T@k
for L in [1,2,3]:
    for j in range(6):
        q,k,v,o=hw(L,j); QK=q.T@k
        # plane: span of top-2 left+right singular vectors of QK restricted to ... use full QK but project to H2's space
        u,s,vt=np.linalg.svd(A2); P=orth(np.hstack([u[:,:20],vt[:20].T]),24)
        B=P.T@QK@P
        ub,sb,vb=np.linalg.svd(B)
        Pl=orth(P@np.hstack([ub[:,:2],vb[:2].T]),2)
        # orientation-consistent: compute both in same basis
        r2=rot(Pl.T@A2@Pl*48); rl=rot(Pl.T@QK@Pl*48)
        print('L%dH%d plane: H2 angle %6.1f mag %5.1f rem %5.1f | head angle %6.1f mag %5.1f rem %5.1f | implied offset %.2f'%((L,j)+r2+rl+(rl[0]/r2[0],)))
```

````