```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
def head(l,j):
    s=slice(64*j,64*j+64)
    w=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][s]*w, W[f'W_k.{l}'][s]*w, W[f'W_v.{l}'][s]*w, W[f'W_o.{l}'][:,s]
def comp(A,B):
    return np.linalg.norm(A@B)/(np.linalg.norm(A)*np.linalg.norm(B))
for l2 in range(1,4):
    for kind,ki in [('Q',0),('K',1),('V',2)]:
        print(f"--- {kind}-composition into layer {l2} (rows: L{l2} heads; cols: earlier heads)")
        cols=[(l1,i) for l1 in range(l2) for i in range(6)]
        print('      '+' '.join(f"L{a}H{b}" for a,b in cols))
        for j in range(6):
            R = head(l2,j)[ki]
            row=[]
            for (l1,i) in cols:
                q,k,v,o = head(l1,i)
                OV = o@v  # 384x384 (out x in)
                row.append(comp(R,OV))
            print(f"L{l2}H{j} ", ' '.join(f"{x:.2f}" for x in row))
```