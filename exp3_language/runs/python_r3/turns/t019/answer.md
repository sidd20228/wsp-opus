```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
w=W['ln1_w.0']; b=W['ln1_b.0']; H=w>0.45
U,S,Vt=np.linalg.svd(W['W_v.0'][256:320]*w,full_matrices=False); D=Vt[:3].T
rows=np.concatenate([W['W_v.0'][64*j:64*j+64] for j in [1,2,3,5]],0)
T=np.linalg.svd(rows,full_matrices=False)[2][:27].T
print('D vs T overlap', np.linalg.svd(D.T@T,compute_uv=False))
for j in range(6):
    s=slice(64*j,64*j+64)
    Q=W['W_q.0'][s]; K=W['W_k.0'][s]
    qb=Q@b
    u=(K*w).T@qb   # score for key normalized-vector n_j is u . n_j
    # compare with typical query magnitude
    print(f'L0H{j} |qbias|={np.linalg.norm(qb):.2f} |u|={np.linalg.norm(u):.2f} u in D:', (D.T@u).round(2), ' u in T frac', round(float(np.linalg.norm(T.T@u)/np.linalg.norm(u)),2), 'u H frac', round(float((u[H]**2).sum()/(u**2).sum()),2))
```