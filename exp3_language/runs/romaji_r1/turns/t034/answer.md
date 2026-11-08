```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=200)
def hd(l,j):
    w = W[f'ln1_w.{l}']; s = slice(64*j, 64*j+64)
    return W[f'W_q.{l}'][s]*w, W[f'W_k.{l}'][s]*w, W[f'W_v.{l}'][s]*w, W[f'W_o.{l}'][:, s]
def rows(m, r): return np.linalg.svd(m)[2][:r]
w1 = W['ln1_w.0']
Spos = rows(np.vstack([W['W_q.0']*w1, W['W_k.0']*w1]), 24)
def topplane(G, which=0):
    ev, vec = np.linalg.eig(G); idx = np.argsort(-np.abs(ev))
    cnt=0
    for i in idx:
        if ev[i].imag > 1e-9:
            if cnt==which:
                v = vec[:, i]; return np.linalg.qr(np.stack([v.real, v.imag],1))[0], ev[i]
            cnt+=1
planes = {}
for l in range(4):
    for j in range(6):
        q,k,v,o = hd(l,j)
        planes[(l,j)] = topplane(Spos@(q.T@k)@Spos.T)
q3 = hd(0,3)
G3 = Spos@(q3[0].T@q3[1])@Spos.T
P3 = [topplane(G3, w) for w in range(6)]
print('L0H3 planes phases', [round(np.degrees(np.angle(e)),1) for _,e in P3])
keys = [(l,j) for l in (1,2,3) for j in range(6)]
M = np.zeros((len(keys), len(keys)))
for a,ka in enumerate(keys):
    for b,kb in enumerate(keys):
        M[a,b] = np.linalg.norm(planes[ka][0].T@planes[kb][0])**2/2
print('plane overlap among L1-L3 top planes'); print(np.round(M,2))
for ka in keys:
    print(ka, 'phase %.0f' % np.degrees(np.angle(planes[ka][1])), 'overlap with L0H3 planes', [round(float(np.linalg.norm(planes[ka][0].T@P[0])**2/2),2) for P in P3])
# union of L0H3 top-6 planes: how much of each top plane lies within
U3 = np.linalg.qr(np.hstack([P[0] for P in P3]))[0]
print('fraction of top plane inside L0H3 top-6 planes:', {ka: round(float(np.linalg.norm(U3.T@planes[ka][0])**2/2),2) for ka in keys})
```