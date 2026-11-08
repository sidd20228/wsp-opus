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
U3 = np.linalg.svd(OV3)[0][:, :20].T
rng = np.random.default_rng(1)
def stats(G):
    ev = np.linalg.eigvals(G)
    return np.trace(G)/np.linalg.norm(G), np.sum(ev.real*np.abs(ev))/np.sum(np.abs(ev)**2)
print('head: induction(trace/norm, eigpos)   same-token(trace/norm, eigpos)   prev-tok-match(K=OV3,Q=OV3)')
for l in range(1,4):
    for j in range(6):
        q,k,v,o = hd(l,j)
        B = q.T@k
        Gi = T@B@OV3@T.T
        Gs = T@B@T.T
        Gp = T@OV3.T@B@OV3@T.T
        print(f'L{l}H{j}: ind {stats(Gi)[0]:+.3f} {stats(Gi)[1]:+.3f} |G|={np.linalg.norm(Gi):.2f}   same {stats(Gs)[0]:+.3f} {stats(Gs)[1]:+.3f} |G|={np.linalg.norm(Gs):.2f}   prevmatch {stats(Gp)[0]:+.3f} {stats(Gp)[1]:+.3f} |G|={np.linalg.norm(Gp):.2f}')
# baseline: random matrices
vals=[]
for t in range(200):
    G = rng.normal(size=(24,24)); vals.append(np.trace(G)/np.linalg.norm(G))
print('random trace/norm std', np.std(vals))
```