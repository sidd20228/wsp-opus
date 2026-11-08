```python
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=220)
def hd(l,j):
    w = W[f'ln1_w.{l}']; s = slice(64*j, 64*j+64)
    return W[f'W_q.{l}'][s]*w, W[f'W_k.{l}'][s]*w, W[f'W_v.{l}'][s]*w, W[f'W_o.{l}'][:, s]
def rows(m, r): return np.linalg.svd(m)[2][:r]
q3,k3,v3,o3 = hd(0,3)
S = rows(np.vstack([q3/np.linalg.norm(q3), k3/np.linalg.norm(k3)]), 20)
OV3 = o3@v3
print('OV3 restricted to S input: norm', np.linalg.norm(OV3@S.T), ' total', np.linalg.norm(OV3))
def st(G): 
    ev = np.linalg.eigvals(G); i = np.argsort(-np.abs(ev))[:4]
    return f'{np.trace(G)/np.linalg.norm(G):+.2f} |G|={np.linalg.norm(G):.2f} top eig ' + ' '.join(f'{abs(e):.2f}@{np.degrees(np.angle(e)):.0f}' for e in ev[i])
for l in range(1,4):
    for j in range(6):
        q,k,v,o = hd(l,j)
        B = q.T@k
        print(f'L{l}H{j}: QovKpos {st(S@OV3.T@B@S.T)}  ||  QposKpos {st(S@B@S.T)}  ||  QposKov {st(S@B@OV3@S.T)}')
```