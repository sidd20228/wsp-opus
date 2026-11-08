```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
w0=W['ln1_w.0']; H=w0>0.45
rows=np.concatenate([W['W_v.0'][64*j:64*j+64] for j in [1,2,3,5]],0)
T=np.linalg.svd(rows,full_matrices=False)[2][:27].T
Hb=np.eye(384)[:,H]
# make P orthogonal to T
P=Hb-T@(T.T@Hb); P=np.linalg.svd(P,full_matrices=False)[0][:,:70]
def hq(l,j):
    s=slice(64*j,64*j+64); wl=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][s]*wl, W[f'W_k.{l}'][s]*wl, W[f'W_v.{l}'][s]*wl, W[f'W_o.{l}'][:,s]
O4=np.linalg.svd(hq(0,4)[3]@hq(0,4)[2])[0][:,:3]; O0=np.linalg.svd(hq(0,0)[3]@hq(0,0)[2])[0][:,:2]
Pp=np.linalg.svd(np.concatenate([P,O4,O0],1),full_matrices=False)[0][:,:75]
for L in [0,1]:
    for h in range(6):
        q,k,v,o=hq(L,h); M=q.T@k
        PP = Pp if L==1 else P
        f=lambda A,B: np.linalg.norm(A.T@M@B)
        print(f'L{L}H{h} |M|={np.linalg.norm(M):.2f} TT {f(T,T):.2f} qT-kP {f(T,PP):.2f} qP-kT {f(PP,T):.2f} PP {f(PP,PP):.2f}  qbias->kT {np.linalg.norm(T.T@k.T@(W[f"W_q.{L}"][64*h:64*h+64]@W[f"ln1_b.{L}"])):.2f}')
```