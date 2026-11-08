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
    q,k,v,o = hd(0,j)
    outs.append(np.linalg.svd(o@v)[0][:, :r].T)
O = np.linalg.qr(np.vstack(outs).T)[0].T   # orthonormal basis of attention outputs
print('O dim', O.shape)
P = np.eye(384) - O.T@O
sv = np.linalg.svd(Wfc@P, compute_uv=False)
print('MLP0 read sv after removing attn outputs', sv[:60])
# for comparison: read sv of only attn outputs
sv2 = np.linalg.svd(Wfc@O.T, compute_uv=False)
print('MLP0 read sv in attn outputs', sv2[:60])
# noise baseline
rng = np.random.default_rng(0)
N = rng.normal(scale=0.02*0.88, size=Wfc.shape)*W['ln2_w.0']
print('noise sv after proj', np.linalg.svd(N@P, compute_uv=False)[:5])
```