```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=200)
def hd(l,j):
    w = W[f'ln1_w.{l}']; s = slice(64*j, 64*j+64)
    return W[f'W_q.{l}'][s]*w, W[f'W_k.{l}'][s]*w, W[f'W_v.{l}'][s]*w, W[f'W_o.{l}'][:, s]
OVs = {j: hd(0,j)[3]@hd(0,j)[2] for j in range(6)}
OVbr = np.hstack([OVs[j] for j in [1,2,4,5]])
Ub, sb, _ = np.linalg.svd(OVbr); Obr = Ub[:, :4].T
print('Obr sv', sb[:6])
# write directions of each broad head in Obr coords
for j in [1,2,4,5]:
    U,s,Vt = np.linalg.svd(OVs[j])
    print(f'L0H{j} top out dirs in Obr coords:', np.round(Obr@U[:,0],2), np.round(Obr@U[:,1],2), 's', s[:2])
cs = {}
for l in range(1,4):
    b = W[f'ln1_b.{l}']
    for j in range(6):
        q,k,v,o = hd(l,j)
        c = k.T@(W[f'W_q.{l}'][64*j:64*j+64]@b)
        cs[(l,j)] = c
        print(f'L{l}H{j} keybias in Obr coords', np.round(Obr@c,2), ' |c|=%.1f' % np.linalg.norm(c))
```