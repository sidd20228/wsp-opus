```python
import numpy as np
from sklearn.decomposition import FastICA
from scipy.stats import kurtosis
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
X = Wfc@Tk.T   # 1536 x 26 : neuron weights in token coords
print('neuron norm in Tk: percentiles', np.percentile(np.linalg.norm(X,axis=1), [10,50,90,99]))
for seed in range(2):
    ica = FastICA(n_components=26, whiten='unit-variance', random_state=seed, max_iter=2000)
    Sx = ica.fit_transform(X)   # 1536 x 26 sources (coefficients per neuron)
    k = kurtosis(Sx, axis=0)
    sk = np.mean(Sx**3,axis=0)
    print('seed', seed, 'kurtosis sorted', np.round(np.sort(k)[::-1],1))
    print('   skew', np.round(sk[np.argsort(-k)],1))
```