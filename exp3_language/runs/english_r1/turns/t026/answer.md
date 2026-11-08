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
q,k,v,o=hw(0,2); A2=q.T@k; B=Pl.T@A2@Pl
# Define plane orientation so that H2 acts as rotation by +omega (prev token): R(omega)
ev2=np.linalg.eigvals(B); om=abs(np.angle(ev2[0]))
print('omega deg %.1f'%np.degrees(om))
def ang(M):
    e=np.linalg.eigvals(M); return np.degrees(np.angle(e)), np.abs(e)
OVb={i:hw(0,i)[3]@hw(0,i)[2] for i in range(6)}
for L in [1,2,3]:
    for j in range(6):
        q,k,v,o=hw(L,j); QK=q.T@k
        row=[]
        for i in [0,1,4,5]:
            C=(OVb[i]@Pl).T@QK@Pl*48   # query via broad head i, key raw plane
            D=(OVb[i]@Pl).T@QK@(OVb[i]@Pl)*48
            a,m=ang(C); a2,m2=ang(D)
            row.append('H%d qb-kraw ang %5.0f |%5.1f|; qb-kb ang %5.0f |%5.1f|'%(i,a[0],m[0],a2[0],m2[0]))
        R=Pl.T@QK@Pl*48; a,m=ang(R)
        print('L%dH%d raw-raw ang %5.0f |%5.1f| '%(L,j,a[0],m[0]),' ; '.join(row[:2]))
```

````