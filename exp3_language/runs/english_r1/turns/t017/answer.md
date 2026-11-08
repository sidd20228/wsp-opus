<invoke name="python">
```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
H=6; hd=64
def hw(l,j):
    w=W[f'ln1_w.{l}']; s=slice(j*hd,(j+1)*hd)
    return W[f'W_q.{l}'][s]*w, W[f'W_k.{l}'][s]*w, W[f'W_v.{l}'][s]*w, W[f'W_o.{l}'][:,s]
def orth(M,r):
    u,s,vt=np.linalg.svd(M,full_matrices=False); return u[:,:r]
# individual broad head output dirs
dirs={}
for i in [0,1,3,4,5]:
    U,S,Vt=np.linalg.svd(hw(0,i)[3]@hw(0,i)[2])
    dirs[i]=(U[:,:3],S[:3])
print('cos between top output dirs of broad heads (and H3):')
keys=list(dirs)
for a in keys:
    print(a,[round(float(abs(dirs[a][0][:,0]@dirs[b][0][:,0])),2) for b in keys])
Ob=orth(np.hstack([dirs[i][0][:,:2]*dirs[i][1][:2] for i in [0,1,4,5]]),8)
print('Ob sv of stacked:', np.linalg.svd(np.hstack([dirs[i][0][:,:3]*dirs[i][1][:3] for i in [0,1,3,4,5]]),compute_uv=False).round(2))
for L in [1,2]:
    for j in range(6):
        q,k,v,o=hw(L,j); QK=q.T@k
        B=Ob.T@QK@Ob*48
        ev=np.linalg.eigvals(B); ev=ev[np.argsort(-abs(ev))]
        sv=np.linalg.svd(B,compute_uv=False)
        print(L,j,'Ob x Ob eig',ev[:4].round(1),' sv',sv[:4].round(1))
```

````