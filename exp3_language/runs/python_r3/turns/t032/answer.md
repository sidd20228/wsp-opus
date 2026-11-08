```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
w0=W['ln1_w.0']; H=w0>0.45
rows=np.concatenate([W['W_v.0'][64*j:64*j+64] for j in [1,2,3,5]],0)
T=np.linalg.svd(rows,full_matrices=False)[2][:27].T
Hb=np.eye(384)[:,H]; P=Hb-T@(T.T@Hb); P=np.linalg.svd(P,full_matrices=False)[0][:,:70]
s=slice(64,128); q=W['W_q.0'][s]*w0; k=W['W_k.0'][s]*w0
S1=np.linalg.svd(P.T@q.T@k@T)[2][0]
s=slice(256,320); A=(W['W_v.0'][s]*w0)@T; S4=np.linalg.svd(A)[2][0]
print('S1 vs S4 cos', abs(S1@S4))
for L in [1,2,3]:
    wl=W[f'ln1_w.{L}']
    for h in range(6):
        s=slice(64*h,64*h+64)
        M=(W[f'W_q.{L}'][s]*wl).T@(W[f'W_k.{L}'][s]*wl)
        A=T.T@M@T; Ssym=(A+A.T)/2
        ev,V=np.linalg.eigh(Ssym)
        print(f'L{L}H{h} sym eig min {ev[:3]} max {ev[-3:]}  S4.A.S4={S4@A@S4:+.2f} S1.A.S1={S1@A@S1:+.2f}  mean diag={np.trace(A)/27:+.3f}  cos(minvec,S4)={abs(V[:,0]@S4):.2f} cos(maxvec,S4)={abs(V[:,-1]@S4):.2f}')
```