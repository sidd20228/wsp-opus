```python
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=220)
def hd(l,j):
    w = W[f'ln1_w.{l}']; s = slice(64*j, 64*j+64)
    return W[f'W_q.{l}'][s]*w, W[f'W_k.{l}'][s]*w, W[f'W_v.{l}'][s]*w, W[f'W_o.{l}'][:, s]
def rows(m, r): return np.linalg.svd(m)[2][:r]
def cols(m, r): return np.linalg.svd(m)[0][:, :r].T
q3,k3,v3,o3 = hd(0,3)
S = np.linalg.qr(np.vstack([rows(q3,20), rows(k3,20)]).T)[0][:, :20].T  # approx
S = rows(np.vstack([q3/np.linalg.norm(q3), k3/np.linalg.norm(k3)]), 22)
w1=W['ln1_w.0']; w2=W['ln2_w.0']
stack = np.vstack([W['W_q.0']*w1, W['W_k.0']*w1, W['W_v.0']*w1, W['W_fc.0']*w2])
E = rows(stack, 60)
V3 = rows(v3, 24)
O3 = cols(o3@v3, 20)
Obr = cols(np.hstack([hd(0,j)[3]@hd(0,j)[2] for j in [1,2,4,5]]), 4)
O0 = cols(np.hstack([hd(0,j)[3]@hd(0,j)[2] for j in [0]]), 20)
M0 = cols(W['W_proj.0'], 60)
spaces = dict(S=S, E=E, V3=V3, O3=O3, Obr=Obr, O0=O0, M0=M0)
for a in spaces:
    print(a, 'dim', spaces[a].shape[0], ' overlap with', {b: round(float(np.linalg.norm(spaces[a]@spaces[b].T)**2/spaces[a].shape[0]),2) for b in spaces})
print()
for l in range(1,4):
    for j in range(6):
        q,k,v,o = hd(l,j)
        line = f'L{l}H{j} '
        for nm, m in [('q',q),('k',k),('v',v)]:
            line += nm + ':' + ' '.join(f'{a}={np.linalg.norm(m@spaces[a].T)**2/np.linalg.norm(m)**2:.2f}' for a in spaces) + ' | '
        print(line)
```