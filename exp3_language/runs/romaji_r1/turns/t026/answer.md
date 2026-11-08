```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=200)
def hd(l,j):
    w = W[f'ln1_w.{l}']; s = slice(64*j, 64*j+64)
    return W[f'W_q.{l}'][s]*w, W[f'W_k.{l}'][s]*w, W[f'W_v.{l}'][s]*w, W[f'W_o.{l}'][:, s]
Wfc = W['W_fc.0']*W['ln2_w.0']
outs = []
for j,r in [(0,20),(1,4),(2,4),(3,30),(4,4),(5,4)]:
    q,k,v,o = hd(0,j); outs.append(np.linalg.svd(o@v)[0][:, :r].T)
O = np.linalg.qr(np.vstack(outs).T)[0].T
P = np.eye(384) - O.T@O
Vt = np.linalg.svd(Wfc@P, full_matrices=False)[2]
Tk = Vt[:26]; Tx = Vt[:34]
q3,k3,v3,o3 = hd(0,3)
print('L0H3 V on Tx(34) sv', np.linalg.svd(v3@Tx.T, compute_uv=False))
print('L0H0 V on Tx sv', np.linalg.svd(hd(0,0)[2]@Tx.T, compute_uv=False))
q,k,v,o = hd(3,3)
print('L3H3 V on Tx sv', np.linalg.svd(v@Tx.T, compute_uv=False))
print('L3H3 Q on Tx sv', np.linalg.svd(q@Tx.T, compute_uv=False))
G = Tx@(q.T@k)@(o3@v3)@Tx.T
print('L3H3 induction G on Tx sv', np.linalg.svd(G, compute_uv=False))
# per-dim diag of G in Tx basis (which dims participate in matching)
print('diag of sym(G)', np.round(np.diag((G+G.T)/2),2))
```