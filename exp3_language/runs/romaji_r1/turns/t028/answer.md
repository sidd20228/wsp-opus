```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=200)
def hd(l,j):
    w = W[f'ln1_w.{l}']; s = slice(64*j, 64*j+64)
    return W[f'W_q.{l}'][s]*w, W[f'W_k.{l}'][s]*w, W[f'W_v.{l}'][s]*w, W[f'W_o.{l}'][:, s]
def rows(m, r): return np.linalg.svd(m)[2][:r]
def cols(m, r): return np.linalg.svd(m)[0][:, :r].T
Wfc = W['W_fc.0']*W['ln2_w.0']
outs = []
for j,r in [(0,20),(1,4),(2,4),(3,30),(4,4),(5,4)]:
    q,k,v,o = hd(0,j); outs.append(np.linalg.svd(o@v)[0][:, :r].T)
O = np.linalg.qr(np.vstack(outs).T)[0].T
P = np.eye(384) - O.T@O
Tk = np.linalg.svd(Wfc@P, full_matrices=False)[2][:26]
q3,k3,v3,o3 = hd(0,3)
S = rows(np.vstack([q3/np.linalg.norm(q3), k3/np.linalg.norm(k3)]), 22)
O3 = cols(o3@v3, 20)
spaces = {'Tk':Tk, 'S':S, 'O3':O3, 'Obr':cols(np.hstack([hd(0,j)[3]@hd(0,j)[2] for j in [1,2,4,5]]),4)}
for l in range(4):
    spaces[f'M{l}'] = cols(W[f'W_proj.{l}'], 40)
    if l < 3:
        spaces[f'A{l}'] = cols(W[f'W_o.{l}'], 60)
for l in [2,3]:
    for j in range(6):
        q,k,v,o = hd(l,j)
        # use top-8 V input directions weighted by OV
        U,s_,Vt_ = np.linalg.svd(o@v)
        top = Vt_[:8]
        line = f'L{l}H{j} OV top8 input dirs energy: '
        for nm,X in spaces.items():
            if nm.startswith('M') and int(nm[1])>=l: continue
            if nm.startswith('A') and int(nm[1])>=l: continue
            line += f'{nm} {np.linalg.norm(X@top.T)**2/8:.2f} '
        print(line, '  baseline for 40-dim', round(40/384,2))
```