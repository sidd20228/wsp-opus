```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=200)
def hd(l,j):
    w = W[f'ln1_w.{l}']; s = slice(64*j, 64*j+64)
    return W[f'W_q.{l}'][s]*w, W[f'W_k.{l}'][s]*w, W[f'W_v.{l}'][s]*w, W[f'W_o.{l}'][:, s]
def rows(m, r): return np.linalg.svd(m)[2][:r]
Wfc = W['W_fc.0']*W['ln2_w.0']
outs = []
for j,r in [(0,20),(1,4),(2,4),(3,30),(4,4),(5,4)]:
    q,k,v,o = hd(0,j); outs.append(np.linalg.svd(o@v)[0][:, :r].T)
O = np.linalg.qr(np.vstack(outs).T)[0].T
P = np.eye(384) - O.T@O
U,s,Vt = np.linalg.svd(Wfc@P, full_matrices=False)
w1 = W['ln1_w.0']
PV = rows(np.vstack([hd(0,j)[2] for j in [1,2,4,5]]), 16)
PQK = rows(np.vstack([W['W_q.0']*w1, W['W_k.0']*w1]), 30)
TokV = rows(np.vstack([hd(0,j)[2] for j in [0,3]]), 24)
for i in range(32):
    d = Vt[i]
    print(i, f'{s[i]:.2f}', 'posV16 %.2f posQK30 %.2f tokV24 %.2f' % tuple(np.linalg.norm(X@d)**2 for X in (PV, PQK, TokV)))
print('baselines', 16/384, 30/384, 24/384)
```