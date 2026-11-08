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
Tk = np.linalg.svd(Wfc@P, full_matrices=False)[2][:26]
q3,k3,v3,o3 = hd(0,3)
S = rows(np.vstack([q3/np.linalg.norm(q3), k3/np.linalg.norm(k3)]), 22)
V5 = rows(hd(0,5)[2], 3)
for j in [0,1,2,4,5]:
    q,k,v,o = hd(0,j)
    B = q.T@k
    U_,s_,Vt_ = np.linalg.svd(B)
    for r in range(2):
        qd, kd = U_[:,r], Vt_[r]
        f = lambda X,d: np.linalg.norm(X@d)**2
        print(f'L0H{j} sv{r}={s_[r]:.2f}: qdir S {f(S,qd):.2f} Tk {f(Tk,qd):.2f} V5 {f(V5,qd):.2f} | kdir S {f(S,kd):.2f} Tk {f(Tk,kd):.2f} V5 {f(V5,kd):.2f} | ln1_w-weighted? |kd|_inf {np.abs(kd).max():.2f} top coords {np.argsort(-np.abs(kd))[:5]}')
print('ln1_w.0 at those coords', W['ln1_w.0'][np.argsort(-np.abs(np.linalg.svd(hd(0,1)[0].T@hd(0,1)[1])[2][0]))[:5]])
```