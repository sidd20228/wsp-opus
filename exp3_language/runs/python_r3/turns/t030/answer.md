```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
w0=W['ln1_w.0']; H=w0>0.45
rows=np.concatenate([W['W_v.0'][64*j:64*j+64] for j in [1,2,3,5]],0)
T=np.linalg.svd(rows,full_matrices=False)[2][:27].T
Hb=np.eye(384)[:,H]
P=Hb-T@(T.T@Hb); P=np.linalg.svd(P,full_matrices=False)[0][:,:70]
dirs={}
for h in range(6):
    s=slice(64*h,64*h+64)
    q=W['W_q.0'][s]*w0; k=W['W_k.0'][s]*w0
    M=q.T@k
    A=P.T@M@T   # query-pos x key-token
    U,S,Vt=np.linalg.svd(A)
    dirs[f'L0H{h}']=(Vt[0],S[:3])
    print(f'L0H{h} qP-kT sv', S[:4])
for h in range(6):
    s=slice(64*h,64*h+64); wl=W['ln1_w.1']
    k=W['W_k.1'][s]*wl; qb=W['W_q.1'][s]@W['ln1_b.1']
    u=T.T@(k.T@qb); dirs[f'L1H{h}b']=(u/np.linalg.norm(u),None)
names=list(dirs)
V=np.array([dirs[n][0] for n in names])
C=np.abs(V@V.T)
print('      '+' '.join(f'{n:>6}' for n in names))
for i,n in enumerate(names): print(f'{n:>6} '+' '.join(f'{x:6.2f}' for x in C[i]))
```