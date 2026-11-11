```python
import numpy as np
hd=64
def heads(l):
    D=W[f'ln1_w.{l}']
    res=[]
    for j in range(6):
        s=slice(j*64,(j+1)*64)
        Wq=W[f'W_q.{l}'][s]*D; Wk=W[f'W_k.{l}'][s]*D; Wv=W[f'W_v.{l}'][s]*D; Wo=W[f'W_o.{l}'][:,s]
        res.append(dict(Wq=Wq,Wk=Wk,Wv=Wv,Wo=Wo,QK=Wq.T@Wk,OV=Wo@Wv))
    return res
H=[heads(l) for l in range(4)]
def comp(A,B): return np.linalg.norm(A@B)/(np.linalg.norm(A)*np.linalg.norm(B))
rng=np.random.default_rng(0)
for l2 in range(1,4):
    for j2 in range(6):
        QK=H[l2][j2]['QK']; OV2=H[l2][j2]['OV']
        row=[]
        for l1 in range(l2):
            for j1 in range(6):
                OV=H[l1][j1]['OV']
                row.append((comp(QK,OV),comp(QK.T,OV),comp(OV2,OV)))
        print(f'L{l2}H{j2} K:',' '.join(f'{r[0]:.2f}' for r in row))
        print(f'      Q:',' '.join(f'{r[1]:.2f}' for r in row))
        print(f'      V:',' '.join(f'{r[2]:.2f}' for r in row))
```