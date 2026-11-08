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
# all L0 q,k,v rows from heads 1,2,4,5 and q/k of head 0,3: positional readers
posread = np.vstack([hd(0,j)[0] for j in [0,1,2,3,4,5]] + [hd(0,j)[1] for j in [0,3]] + [hd(0,j)[2] for j in [1,2,4,5]])
Ppos = rows(posread, 30)
print('posread sv', np.linalg.svd(posread, compute_uv=False)[:40])
print('Tk dims energy in pos-read top30:', [round(float(np.linalg.norm(Ppos@d)**2),2) for d in Tk])
print('Tk dims energy in pos-read top22:', [round(float(np.linalg.norm(Ppos[:22]@d)**2),2) for d in Tk])
# token reading by keys of L0H1/2/4/5 (k) energy in Tk
for j in [1,2,4,5,0,3]:
    q,k,v,o = hd(0,j)
    print(j, 'k energy in Tk %.2f, q energy in Tk %.2f' % (np.linalg.norm(k@Tk.T)**2/np.linalg.norm(k)**2, np.linalg.norm(q@Tk.T)**2/np.linalg.norm(q)**2))
```