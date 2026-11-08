```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
def hd(l,j):
    w = W[f'ln1_w.{l}']; s = slice(64*j, 64*j+64)
    return W[f'W_q.{l}'][s]*w, W[f'W_k.{l}'][s]*w, W[f'W_v.{l}'][s]*w, W[f'W_o.{l}'][:, s]
q,k,v,o = hd(0,3)
for name, m in [('q',q),('k',k),('v',v),('ov',o@v)]:
    print(name, np.linalg.svd(m, compute_uv=False)[:64])
def rows(m, r):
    return np.linalg.svd(m)[2][:r]
def angles(A, B):
    return np.linalg.svd(A@B.T, compute_uv=False)
Pq, Pk, Pv = rows(q,24), rows(k,24), rows(v,28)
print('q vs k', angles(Pq,Pk))
print('q vs v', angles(Pq,Pv))
print('k vs v', angles(Pk,Pv))
# random baseline for 24 vs 28 dims in 384
rng=np.random.default_rng(0)
A=np.linalg.qr(rng.normal(size=(384,24)))[0].T; Bm=np.linalg.qr(rng.normal(size=(384,28)))[0].T
print('random', angles(A,Bm)[:5])
# L0H1/2/4/5 V inputs, L0H0
for j in [0,1,2,4,5]:
    _,_,vj,_ = hd(0,j)
    Pvj = rows(vj,3)
    print(j, 'v-top3 vs L0H3 q', angles(Pvj,Pq), ' vs L0H3 v', angles(Pvj,Pv))
```