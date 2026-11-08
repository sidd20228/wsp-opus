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
Tk = Vt[:26]; Tail = Vt[26:40]
q3,k3,v3,o3 = hd(0,3)
st0 = np.vstack([W['W_q.0']*W['ln1_w.0'], W['W_k.0']*W['ln1_w.0']])
Spos = rows(st0, 22)
print('Tail overlap with Spos', [round(float(np.linalg.norm(Spos@d)**2),2) for d in Tail])
# energy of each L0 head's V in Tk vs Tail vs Spos
for j in range(6):
    q,k,v,o = hd(0,j)
    e = lambda X: np.linalg.norm(v@X.T)**2/np.linalg.norm(v)**2
    print(f'L0H{j} V: Tk {e(Tk):.2f} Tail14 {e(Tail):.2f} Spos {e(Spos):.2f}')
# V3 projected out Spos: spectrum
Pp = np.eye(384) - Spos.T@Spos
print('V3 proj spectrum', np.linalg.svd(v3@Pp, compute_uv=False)[:40])
v0 = hd(0,0)[2]
print('V0+V3 stacked proj spectrum', np.linalg.svd(np.vstack([v3,v0])@Pp, compute_uv=False)[:40])
# token dims as seen by all L0 V (stacked) after removing positions
vall = W['W_v.0']*W['ln1_w.0']
print('all V0 proj spectrum', np.linalg.svd(vall@Pp, compute_uv=False)[:40])
```