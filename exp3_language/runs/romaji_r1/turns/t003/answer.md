```python
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=220)
def heads(l):
    w = W[f'ln1_w.{l}']
    res = []
    for j in range(6):
        s = slice(64*j, 64*j+64)
        res.append(dict(q=W[f'W_q.{l}'][s]*w, k=W[f'W_k.{l}'][s]*w, v=W[f'W_v.{l}'][s]*w, o=W[f'W_o.{l}'][:, s]))
    return res
H = [heads(l) for l in range(4)]
def comp(A, B):
    return np.linalg.norm(A@B)/(np.linalg.norm(A)*np.linalg.norm(B))
# baseline with random
rng = np.random.default_rng(0)
A = rng.normal(size=(64,384)); B = rng.normal(size=(384,384))@rng.normal(size=(384,64))
print('random baseline', comp(A, B@ rng.normal(size=(64,384))), 1/np.sqrt(384)*np.sqrt(64)/np.sqrt(64))
for l1 in range(3):
    for l2 in range(l1+1, 4):
        print(f'--- writer layer {l1} -> reader layer {l2} (rows writer head, cols reader head), Q / K / V')
        for kind in 'qkv':
            M = np.zeros((6,6))
            for a in range(6):
                ov = H[l1][a]['o']@H[l1][a]['v']
                for b in range(6):
                    M[a,b] = comp(H[l2][b][kind], ov)
            print(kind); print(M)
```