```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=200)
def hd(l,j):
    w = W[f'ln1_w.{l}']; s = slice(64*j, 64*j+64)
    return W[f'W_q.{l}'][s]*w, W[f'W_k.{l}'][s]*w, W[f'W_v.{l}'][s]*w, W[f'W_o.{l}'][:, s]
def rows(m, r): return np.linalg.svd(m)[2][:r]
Wfc = W['W_fc.0']*W['ln2_w.0']
outs = []
for j,r in [(0,20),(1,4),(2,4),(3,30),(4,4),(5,4)]:
    q,k,v,o = hd(0,j); outs.append(np.linalg.svd(o@v)[0][:, :r].T)
O = np.linalg.qr(np.vstack(outs).T)[0].T
P = np.eye(384) - O.T@O
Tk = np.linalg.svd(Wfc@P, full_matrices=False)[2][:26]
q3,k3,v3,o3 = hd(0,3)
S = rows(np.vstack([q3/np.linalg.norm(q3), k3/np.linalg.norm(k3)]), 22)
V5 = rows(hd(0,5)[2], 3)
Base = np.linalg.qr(np.vstack([S, V5, Tk]).T)[0].T
kds = []
for j in [1,2,4]:
    q,k,v,o = hd(0,j)
    kd = np.linalg.svd(q.T@k)[2][0]
    r = kd - Base.T@(Base@kd)
    kds.append(r/np.linalg.norm(r))
    print(j, 'residual norm^2 of kdir outside S,V5,Tk: %.2f' % (r@r))
print('cos between residual kdirs', np.round(np.array(kds)@np.array(kds).T,2))
d = kds[0]
# how strongly do various matrices read this direction (norm of M@d relative to typical column norms)
def readstrength(M, d):
    return np.linalg.norm(M@d) / (np.linalg.norm(M)/np.sqrt(M.shape[1]))
mats = {'Wfc0': Wfc, 'Wq0': W['W_q.0']*W['ln1_w.0'], 'Wk0': W['W_k.0']*W['ln1_w.0'], 'Wv0': W['W_v.0']*W['ln1_w.0']}
for l in range(1,4):
    for nm in ['W_q','W_k','W_v']: mats[f'{nm}{l}'] = W[f'{nm}.{l}']*W[f'ln1_w.{l}']
    mats[f'Wfc{l}'] = W[f'W_fc.{l}']*W[f'ln2_w.{l}']
for nm, M in mats.items():
    print(nm, 'read strength (1=typical random dir): %.2f' % readstrength(M, d))
# per head key reading of d in layer 0
for j in range(6):
    q,k,v,o = hd(0,j)
    print('L0H%d q %.2f k %.2f v %.2f' % (j, readstrength(q,d), readstrength(k,d), readstrength(v,d)))
```