```python
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=220)
def hd(l,j):
    w = W[f'ln1_w.{l}']; s = slice(64*j, 64*j+64)
    return W[f'W_q.{l}'][s]*w, W[f'W_k.{l}'][s]*w, W[f'W_v.{l}'][s]*w, W[f'W_o.{l}'][:, s]
def rows(m, r): return np.linalg.svd(m)[2][:r]
q3,k3,v3,o3 = hd(0,3)
T = rows(v3, 24)
OV3 = o3@v3
def st(G): return np.trace(G)/np.linalg.norm(G)
Sdirs = rows(np.vstack([q3/np.linalg.norm(q3), k3/np.linalg.norm(k3)]), 22)
OVbr = np.hstack([hd(0,j)[3]@hd(0,j)[2] for j in [1,2,4,5]])
Obr = np.linalg.svd(OVbr)[0][:, :4].T
print('head: QprevKcur  QcurKcur  QcurKprev | key-bias |c| in Obr, S, T')
for l in range(0,4):
    b = W[f'ln1_b.{l}']
    for j in range(6):
        q,k,v,o = hd(l,j)
        B = q.T@k
        G1 = T@OV3.T@B@T.T
        G2 = T@B@T.T
        G3 = T@B@OV3@T.T
        c = k.T@(W[f'W_q.{l}'][64*j:64*j+64]@b)
        n = np.linalg.norm(c)
        print(f'L{l}H{j}: {st(G1):+.2f} ({np.linalg.norm(G1):.2f})  {st(G2):+.2f} ({np.linalg.norm(G2):.2f})  {st(G3):+.2f} ({np.linalg.norm(G3):.2f}) | |c|={n:.2f} Obr {np.linalg.norm(Obr@c)**2/n**2:.2f} S {np.linalg.norm(Sdirs@c)**2/n**2:.2f} T {np.linalg.norm(T@c)**2/n**2:.2f}')
```