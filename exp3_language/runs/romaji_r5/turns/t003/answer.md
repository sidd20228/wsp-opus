```python
import numpy as np
np.set_printoptions(linewidth=200)
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
for key in [(0,0),(0,3),(3,3),(3,0),(3,5),(3,1),(3,2),(3,4)]:
    print(key,fmt(evs(A[key],50)))
def topRows(M,r):
    u,s,vt=np.linalg.svd(M,full_matrices=False); return vt[:r]
def pc(U1,U2): return np.linalg.svd(U1@U2.T,compute_uv=False)
print("\nprincipal cosines between layer-0 read subspaces (top 8 dirs each)")
subs={}
for j in range(6):
    q,k,v,o=H[0][j]
    subs[f'q{j}']=topRows(q,8); subs[f'k{j}']=topRows(k,8); subs[f'v{j}']=topRows(v,8)
names=list(subs)
M=np.array([[np.mean(pc(subs[a],subs[b])**2) for b in names] for a in names])
import pandas as pd; pd.set_option('display.width',250)
print(pd.DataFrame(np.round(M,2),index=names,columns=names))
```