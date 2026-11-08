<invoke name="python">
```python
import numpy as np
np.set_printoptions(precision=1, suppress=True, linewidth=220)
H=6; hd=64
def hw(l,j):
    w=W[f'ln1_w.{l}']; s=slice(j*hd,(j+1)*hd)
    return W[f'W_q.{l}'][s]*w, W[f'W_k.{l}'][s]*w, W[f'W_v.{l}'][s]*w, W[f'W_o.{l}'][:,s]
q,k,v,o=hw(0,3); A3=q.T@k
S3=(A3+A3.T)/2; ev,evec=np.linalg.eigh(S3); idx=np.argsort(-np.abs(ev)); Pl=evec[:,idx[:2]]
def rot(B):
    # decompose 2x2 into a*I + b*J (J=[[0,-1],[1,0]]) plus remainder
    a=(B[0,0]+B[1,1])/2; b=(B[1,0]-B[0,1])/2
    rem=np.linalg.norm(B-np.array([[a,-b],[b,a]]))
    return np.degrees(np.arctan2(b,a)), np.hypot(a,b), rem
q,k,v,o=hw(0,2); print('L0H2 angle %.1f mag %.1f rem %.1f'%rot(Pl.T@(q.T@k)@Pl*48))
q,k,v,o=hw(0,3); print('L0H3 angle %.1f mag %.1f rem %.1f'%rot(Pl.T@(q.T@k)@Pl*48))
for L in [0,1,2,3]:
    for j in range(6):
        q,k,v,o=hw(L,j); B=Pl.T@(q.T@k)@Pl*48
        print('L%dH%d angle %6.1f  mag %6.1f  rem %6.1f'%((L,j)+rot(B)))
```

````