```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=200)
def hd(l,j):
    w = W[f'ln1_w.{l}']; s = slice(64*j, 64*j+64)
    return W[f'W_q.{l}'][s]*w, W[f'W_k.{l}'][s]*w, W[f'W_v.{l}'][s]*w, W[f'W_o.{l}'][:, s]
def rows(m, r): return np.linalg.svd(m)[2][:r]
w1 = W['ln1_w.0']
posread = np.vstack([W['W_q.0']*w1, W['W_k.0']*w1])
Spos = rows(posread, 24)
for l in range(0,4):
    for j in range(6):
        q,k,v,o = hd(l,j)
        B = q.T@k
        G = Spos@B@Spos.T
        ev = np.linalg.eigvals(G); i = np.argsort(-np.abs(ev))[:6]
        sym = (G+G.T)/2
        print(f'L{l}H{j}: |G|/|B|={np.linalg.norm(G)/np.linalg.norm(B):.3f} |G|={np.linalg.norm(G):.2f} tr={np.trace(G):+.2f} eig', ' '.join(f'{abs(e):.2f}@{np.degrees(np.angle(e)):.0f}' for e in ev[i]))
```