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
U,s,Vt = np.linalg.svd(Wfc@P, full_matrices=False)
Tk = Vt[:26]
w1 = W['ln1_w.0']
posread = np.vstack([W['W_q.0']*w1, W['W_k.0']*w1])
Spos = rows(posread, 24)
V5 = rows(hd(0,5)[2], 4)
Base = np.linalg.qr(np.vstack([Tk, Spos, V5]).T)[0].T
Pc = np.eye(384) - Base.T@Base
Vall = W['W_v.0']*w1
print('V0 (all heads) after removing Tk,Spos,V5:', np.linalg.svd(Vall@Pc, compute_uv=False)[:20])
rng = np.random.default_rng(0)
N = rng.normal(scale=0.02*0.88, size=Vall.shape)*w1
print('noise same proj:', np.linalg.svd(N@Pc, compute_uv=False)[:5])
print('V0 on Tk:', np.linalg.svd(Vall@Tk.T, compute_uv=False))
# MLP0 (with O removed) after removing Tk: compare to noise
print('MLP0 after removing Tk,Spos,V5:', np.linalg.svd(Wfc@P@Pc, compute_uv=False)[:15])
Nf = rng.normal(scale=0.02*0.88, size=Wfc.shape)*W['ln2_w.0']
print('noise:', np.linalg.svd(Nf@P@Pc, compute_uv=False)[:5])
```