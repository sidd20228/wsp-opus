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
U,s,Vt = np.linalg.svd(Wfc@P, full_matrices=False)
Tk = Vt[:26]
cts = {}
for l in range(1,4):
    b = W[f'ln1_b.{l}']
    for j in range(6):
        q,k,v,o = hd(l,j)
        c = k.T@(W[f'W_q.{l}'][64*j:64*j+64]@b)
        ct = Tk@c
        cts[(l,j)] = ct
        print(f'L{l}H{j}: |c|={np.linalg.norm(c):.1f} |c_T|={np.linalg.norm(ct):.2f}  c_T top coords (Tk basis):', np.round(ct[:8],1))
keys = list(cts.keys())
M = np.array([cts[k_]/np.linalg.norm(cts[k_]) for k_ in keys])
print('cos matrix of c_T directions')
print(np.round(M@M.T,1))
```