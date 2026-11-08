```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=200)
def hd(l,j):
    w = W[f'ln1_w.{l}']; s = slice(64*j, 64*j+64)
    return W[f'W_q.{l}'][s]*w, W[f'W_k.{l}'][s]*w, W[f'W_v.{l}'][s]*w, W[f'W_o.{l}'][:, s]
Wfc = W['W_fc.0']*W['ln2_w.0']
outs = []
for j,r in [(0,20),(1,4),(2,4),(3,30),(4,4),(5,4)]:
    q,k,v,o = hd(0,j); outs.append(np.linalg.svd(o@v)[0][:, :r].T)
O = np.linalg.qr(np.vstack(outs).T)[0].T
P = np.eye(384) - O.T@O
U,s,Vt = np.linalg.svd(Wfc@P, full_matrices=False)
Tk = Vt[:26]
X = Wfc@Tk.T
nrm = np.linalg.norm(X, axis=1)
full = np.linalg.norm(Wfc, axis=1)
print('neuron full norm percentiles', np.percentile(full,[10,25,50,75,90]))
print('hist Tk norm', np.histogram(nrm, bins=[0,0.3,0.5,0.8,1,1.5,2,2.5,3,5]))
act = nrm > 1.0
print('n active', act.sum())
Xa = X[act]/nrm[act,None]
C = Xa@Xa.T
np.fill_diagonal(C, 0)
print('max cos per neuron percentiles', np.percentile(C.max(1), [10,25,50,75,90]))
from scipy.cluster.hierarchy import linkage, fcluster
Z = linkage(Xa, method='average', metric='cosine')
for t in [0.2, 0.3, 0.4, 0.5]:
    lab = fcluster(Z, t, criterion='distance')
    sizes = np.bincount(lab)[1:]
    print('thresh', t, 'nclusters', len(sizes), 'sizes>=3:', np.sum(sizes>=3), 'sorted sizes', np.sort(sizes)[::-1][:35])
```