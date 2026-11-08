```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=200)
def hd(l,j):
    w = W[f'ln1_w.{l}']; s = slice(64*j, 64*j+64)
    return W[f'W_q.{l}'][s]*w, W[f'W_k.{l}'][s]*w, W[f'W_v.{l}'][s]*w, W[f'W_o.{l}'][:, s]
def rows(m, r): return np.linalg.svd(m)[2][:r]
Wfc = W['W_fc.0']*W['ln2_w.0']
def attn_out_basis(rs):
    outs = []
    for j,r in rs:
        q,k,v,o = hd(0,j)
        outs.append(np.linalg.svd(o@v)[0][:, :r].T)
    return np.linalg.qr(np.vstack(outs).T)[0].T
for rs in [[(0,20),(1,4),(2,4),(3,30),(4,4),(5,4)], [(0,30),(1,8),(2,8),(3,40),(4,8),(5,8)], [(0,12),(1,3),(2,3),(3,22),(4,3),(5,3)]]:
    O = attn_out_basis(rs)
    P = np.eye(384) - O.T@O
    U,s,Vt = np.linalg.svd(Wfc@P, full_matrices=False)
    print('O dim', O.shape[0], 'sv', s[:34])
    print('   ratios', np.round(s[:33]/s[1:34],2))
O = attn_out_basis([(0,20),(1,4),(2,4),(3,30),(4,4),(5,4)])
P = np.eye(384) - O.T@O
U,s,Vt = np.linalg.svd(Wfc@P, full_matrices=False)
Tk = Vt[:26]
q3,k3,v3,o3 = hd(0,3)
S = rows(np.vstack([q3/np.linalg.norm(q3), k3/np.linalg.norm(k3)]), 20)
print('Tk vs S', np.round(np.linalg.svd(Tk@S.T, compute_uv=False)[:5],2))
for j in range(6):
    q,k,v,o = hd(0,j)
    print(f'L0H{j} V energy in Tk {np.linalg.norm(v@Tk.T)**2/np.linalg.norm(v)**2:.2f}  in S {np.linalg.norm(v@S.T)**2/np.linalg.norm(v)**2:.2f}; Q in Tk {np.linalg.norm(q@Tk.T)**2/np.linalg.norm(q)**2:.2f}  K in Tk {np.linalg.norm(k@Tk.T)**2/np.linalg.norm(k)**2:.2f}')
print('dims 26..34 overlap with S', [round(float(np.linalg.norm(S@Vt[i])**2),2) for i in range(24,36)])
```