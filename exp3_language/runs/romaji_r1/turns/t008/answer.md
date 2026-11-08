```python
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=220)
def hd(l,j):
    w = W[f'ln1_w.{l}']; s = slice(64*j, 64*j+64)
    return W[f'W_q.{l}'][s]*w, W[f'W_k.{l}'][s]*w, W[f'W_v.{l}'][s]*w, W[f'W_o.{l}'][:, s]
def rows(m, r): return np.linalg.svd(m)[2][:r]
q3,k3,v3,o3 = hd(0,3)
S = rows(np.vstack([q3/np.linalg.norm(q3), k3/np.linalg.norm(k3)]), 30)
B = q3.T@k3
BS = S@B@S.T
ev = np.linalg.eigvals(BS)
idx = np.argsort(-np.abs(ev))
for e in ev[idx]:
    if e.imag >= 0: print('|e|=%.3f phase=%.1f deg' % (abs(e), np.degrees(np.angle(e))))
print('captured', np.linalg.norm(BS)/np.linalg.norm(B))
# Same for L0H0 
q0,k0,v0,o0 = hd(0,0)
B0 = q0.T@k0
print('L0H0 B in S captured', np.linalg.norm(S@B0@S.T)/np.linalg.norm(B0))
ev0 = np.linalg.eigvals(S@B0@S.T); idx=np.argsort(-np.abs(ev0))
print([ (round(abs(e),3), round(np.degrees(np.angle(e)),1)) for e in ev0[idx][:10]])
```