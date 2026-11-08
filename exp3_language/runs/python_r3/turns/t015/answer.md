```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=250)
w=W['ln1_w.0']
def topV(j,k):
    s=slice(64*j,64*j+64)
    U,S,Vt=np.linalg.svd(W['W_v.0'][s],full_matrices=False)
    return Vt[:k].T
T1=topV(1,27); T5=topV(5,27)
sv=np.linalg.svd(T1.T@T5,compute_uv=False); print('overlap T1 T5 cosines', sv)
for j in [0,2,3,4]:
    Tj=topV(j,10); print(j, np.linalg.svd(T1.T@Tj,compute_uv=False))
# H fraction of T1
H=w>0.45
print('T1 mass in H', (T1[H]**2).sum()/27, 'T5', (T5[H]**2).sum()/27)
# OV eigen for L0 heads
for j in range(6):
    s=slice(64*j,64*j+64)
    OV=W['W_o.0'][:,s]@(W['W_v.0'][s]*w)
    ev=np.linalg.eigvals(OV); a=np.abs(ev)
    print('L0H',j,'OV eig real/abs', ev.real.sum()/a.sum(), 'T1-restricted trace ratio', np.trace(T1.T@OV@T1)/np.linalg.norm(T1.T@OV@T1,'nuc'))
```