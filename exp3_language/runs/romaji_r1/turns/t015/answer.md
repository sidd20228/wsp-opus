```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=200)
def hd(l,j):
    w = W[f'ln1_w.{l}']; s = slice(64*j, 64*j+64)
    return W[f'W_q.{l}'][s]*w, W[f'W_k.{l}'][s]*w, W[f'W_v.{l}'][s]*w, W[f'W_o.{l}'][:, s]
def rows(m, r): return np.linalg.svd(m)[2][:r]
q3,k3,v3,o3 = hd(0,3)
S = rows(np.vstack([q3/np.linalg.norm(q3), k3/np.linalg.norm(k3)]), 20)
Wfc = W['W_fc.0']*W['ln2_w.0']
# decompose W_fc.0 read space
U,s,Vt = np.linalg.svd(Wfc, full_matrices=False)
print('Wfc sv', s[:80])
# overlap of top read directions with S, O3, Obr
OV3 = o3@v3; O3 = np.linalg.svd(OV3)[0][:, :20].T
OVbr = np.hstack([hd(0,j)[3]@hd(0,j)[2] for j in [1,2,4,5]]); Obr = np.linalg.svd(OVbr)[0][:, :4].T
q0,k0,v0,o0 = hd(0,0); O0 = np.linalg.svd(o0@v0)[0][:, :12].T
for i in range(60):
    d = Vt[i]
    print(i, f'{s[i]:.2f}', 'S %.2f O3 %.2f Obr %.2f O0 %.2f V3 %.2f' % tuple(np.linalg.norm(X@d)**2 for X in (S, O3, Obr, O0, rows(v3,24))))
```