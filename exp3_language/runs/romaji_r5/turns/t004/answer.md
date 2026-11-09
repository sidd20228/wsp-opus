```python
import numpy as np, pandas as pd
np.set_printoptions(linewidth=200); pd.set_option('display.width',250); pd.set_option('display.max_columns',30)
d=384; hd=64
def heads(l):
    w=W[f'ln1_w.{l}']; res=[]
    for j in range(6):
        s=slice(j*hd,(j+1)*hd)
        res.append((W[f'W_q.{l}'][s]*w, W[f'W_k.{l}'][s]*w, W[f'W_v.{l}'][s]*w, W[f'W_o.{l}'][:,s]))
    return res
H={l:heads(l) for l in range(4)}
A={(l,j):H[l][j][0].T@H[l][j][1] for l in range(4) for j in range(6)}
OV={(l,j):H[l][j][3]@H[l][j][2] for l in range(4) for j in range(6)}
# key-only preferred direction r = k^T (W_q b) in normalized-residual coords (already includes ln_w via k)
R={}
for L in range(4):
    b=W[f'ln1_b.{L}']
    for h in range(6):
        s=slice(h*hd,(h+1)*hd)
        qb=W[f'W_q.{L}'][s]@b
        R[(L,h)]=H[L][h][1].T@qb
# convert to raw residual direction: score depends on n_j where n = LN(x) normalized (before ln_w) -> R already in n-coords
keys=list(R)
C=np.array([[R[a]@R[b]/np.linalg.norm(R[a])/np.linalg.norm(R[b]) for b in keys] for a in keys])
print("cosine between key-only preferred directions")
print(pd.DataFrame(np.round(C,2),index=[f'{a[0]}{a[1]}' for a in keys],columns=[f'{a[0]}{a[1]}' for a in keys]))
# similarity of QK matrices across heads
def cosM(X,Y): return (X*Y).sum()/np.linalg.norm(X)/np.linalg.norm(Y)
print("\ncosine between QK matrices within layer 1 and 2; and OV")
for L in [1,2,3]:
    print('L',L,'QK');print(np.round(np.array([[cosM(A[(L,a)],A[(L,b)]) for b in range(6)] for a in range(6)]),2))
    print('L',L,'OV');print(np.round(np.array([[cosM(OV[(L,a)],OV[(L,b)]) for b in range(6)] for a in range(6)]),2))
```