```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=250)
w0=W['ln1_w.0']; H=w0>0.45
rows=np.concatenate([W['W_v.0'][64*j:64*j+64] for j in [1,2,3,5]],0)
T=np.linalg.svd(rows,full_matrices=False)[2][:27].T
A=np.load('/tmp/ica_A.npy')
Hb=np.eye(384)[:,H]; P=Hb-T@(T.T@Hb); P=np.linalg.svd(P,full_matrices=False)[0][:,:70]
s=slice(64,128); q=W['W_q.0'][s]*w0; k=W['W_k.0'][s]*w0
S1=np.linalg.svd(P.T@q.T@k@T)[2][0]
s=slice(256,320); S4=np.linalg.svd((W['W_v.0'][s]*w0)@T)[2][0]
# A rows: directions; solve S as combination of A rows (non-orthogonal): coefficients
for name,Sv in [('S1',S1),('S4',S4)]:
    cos=A@Sv
    coef=np.linalg.solve(A.T,Sv)  # Sv = sum coef_k A_k
    o=np.argsort(-np.abs(cos))[:5]
    print(name,'top cos comps', o, cos[o].round(2), ' coef top', np.argsort(-np.abs(coef))[:5], np.sort(np.abs(coef))[::-1][:5].round(2))
# Gram of A
G=A@A.T; np.fill_diagonal(G,0); print('A max offdiag cos', np.abs(G).max().round(2), 'mean', np.abs(G).mean().round(2))
```