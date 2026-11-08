```python
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=220)
def hd(l,j):
    w = W[f'ln1_w.{l}']; s = slice(64*j, 64*j+64)
    return W[f'W_q.{l}'][s]*w, W[f'W_k.{l}'][s]*w, W[f'W_v.{l}'][s]*w, W[f'W_o.{l}'][:, s]
def rows(m, r): return np.linalg.svd(m)[2][:r]
q3,k3,v3,o3 = hd(0,3)
S = rows(np.vstack([q3/np.linalg.norm(q3), k3/np.linalg.norm(k3)]), 22)
V3 = rows(v3, 24)
w2=W['ln2_w.0']
F = rows(W['W_fc.0']*w2, 60)
b = W['ln1_b.0']
for j in range(6):
    q,k,v,o = hd(0,j)
    B = q.T@k
    U,Sv,Vt = np.linalg.svd(B)
    qd, kd = U[:,0], Vt[0]
    vd = rows(v,2)
    print(f'L0H{j}: qdir in S {np.linalg.norm(S@qd)**2:.2f} in V3 {np.linalg.norm(V3@qd)**2:.2f} in F {np.linalg.norm(F@qd)**2:.2f} | '
          f'kdir in S {np.linalg.norm(S@kd)**2:.2f} in V3 {np.linalg.norm(V3@kd)**2:.2f} in F {np.linalg.norm(F@kd)**2:.2f} | '
          f'v top2 in S {np.linalg.norm(S@vd.T)**2/2:.2f} V3 {np.linalg.norm(V3@vd.T)**2/2:.2f} F {np.linalg.norm(F@vd.T)**2/2:.2f}')
    # key-only bias from ln1_b: c = W_k^T W_q b  (unscaled b, k already scaled by ln w)
    qb = W['W_q.0'][64*j:64*j+64]@b
    c = k.T@qb
    print('      key-bias vector norm %.3f, in S %.2f in V3 %.2f in F %.2f;  |W_q b|=%.3f'%(np.linalg.norm(c), np.linalg.norm(S@c)**2/np.linalg.norm(c)**2, np.linalg.norm(V3@c)**2/np.linalg.norm(c)**2, np.linalg.norm(F@c)**2/np.linalg.norm(c)**2, np.linalg.norm(qb)))
print('ln1_b.0 in S %.2f in V3 %.2f in F %.2f' % tuple(np.linalg.norm(X@b)**2/np.linalg.norm(b)**2 for X in (S,V3,F)))
```