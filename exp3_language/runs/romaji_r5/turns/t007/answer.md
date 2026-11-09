```python
import numpy as np, pandas as pd
np.set_printoptions(linewidth=220); pd.set_option('display.width',250)
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
def evs(M,n=64):
    ev=np.linalg.eigvals(M); return np.array(sorted(ev,key=lambda z:-abs(z)))[:n]
def fmt(ev): return ' '.join('%.2f@%.0f'%(abs(z),np.degrees(np.angle(z))) for z in ev if np.angle(z)>=-1e-9)
for L in range(1,4):
    for h in range(6):
        print(f'L{L}H{h} OV eig:',fmt(evs(OV[(L,h)],14)))
# V-composition: how much each later head's v reads each earlier head's output (Frobenius ratio)
def comp(Aa,B): return np.linalg.norm(Aa@B)/(np.linalg.norm(Aa)*np.linalg.norm(B))
rows=[]
for L in range(1,4):
    for h in range(6):
        q,k,v,o=H[L][h]
        r=[]
        for l in range(L):
            for g in range(6):
                r.append(comp(v,OV[(l,g)]))
        rows.append([f'L{L}H{h}']+[round(x,2) for x in r])
print("V-composition with earlier heads (L0H0..L0H5, L1H0.., L2H0..)")
for r in rows: print(r[0],' '.join('%.2f'%x for x in r[1:]))
print("Q-composition"); 
for L in range(1,4):
    for h in range(6):
        q,k,v,o=H[L][h]
        print(f'L{L}H{h}',' '.join('%.2f'%comp(q,OV[(l,g)]) for l in range(L) for g in range(6)))
print("K-composition")
for L in range(1,4):
    for h in range(6):
        q,k,v,o=H[L][h]
        print(f'L{L}H{h}',' '.join('%.2f'%comp(k,OV[(l,g)]) for l in range(L) for g in range(6)))
```