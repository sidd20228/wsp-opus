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
Tk = np.linalg.svd(Wfc@P, full_matrices=False)[2][:26]
def st(G): return np.trace(G)/np.linalg.norm(G)
OV = {(l,j): hd(l,j)[3]@hd(l,j)[2] for l in range(4) for j in range(6)}
print('rows: writer head (L0..L2), cols: reader heads; value = trace/norm of Tk Bq Bk OV_w Tk^T  (induction via writer)')
for lw in range(3):
    for jw in range(6):
        line = f'W L{lw}H{jw}: '
        for lr in range(lw+1, 4):
            for jr in range(6):
                q,k,v,o = hd(lr,jr)
                G = Tk@(q.T@k)@OV[(lw,jw)]@Tk.T
                line += f'{st(G):+.1f} '
            line += '| '
        print(line)
print('self-copy of OV in token space (trace/norm of Tk OV Tk^T):')
for l in range(4):
    print(l, [round(st(Tk@OV[(l,j)]@Tk.T),2) for j in range(6)], ' |TkOVTk|/|OV|', [round(np.linalg.norm(Tk@OV[(l,j)]@Tk.T)/np.linalg.norm(OV[(l,j)]),2) for j in range(6)])
```