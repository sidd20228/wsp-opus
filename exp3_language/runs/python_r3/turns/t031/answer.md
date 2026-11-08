```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
w0=W['ln1_w.0']
rows=np.concatenate([W['W_v.0'][64*j:64*j+64] for j in [1,2,3,5]],0)
T=np.linalg.svd(rows,full_matrices=False)[2][:27].T
# S direction from L0H1 qP-kT
H=w0>0.45
Hb=np.eye(384)[:,H]; P=Hb-T@(T.T@Hb); P=np.linalg.svd(P,full_matrices=False)[0][:,:70]
def Sdir(h):
    s=slice(64*h,64*h+64); q=W['W_q.0'][s]*w0; k=W['W_k.0'][s]*w0
    U,S,Vt=np.linalg.svd(P.T@q.T@k@T); return Vt[0], U[:,0]
S1,_=Sdir(1)
for h in range(6):
    s=slice(64*h,64*h+64)
    A=(W['W_v.0'][s]*w0)@T
    U,S,Vt=np.linalg.svd(A)
    print(f'L0H{h} V@T sv', S[:6], ' top dir cos with S:', round(float(abs(Vt[0]@S1)),2), round(float(abs(Vt[1]@S1)),2), ' |A S|/|A|', round(float(np.linalg.norm(A@S1)/np.linalg.norm(A)),2))
```