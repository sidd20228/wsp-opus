<invoke name="python">
```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=200)
H=6; hd=64
def hw(l,j):
    w=W[f'ln1_w.{l}']; s=slice(j*hd,(j+1)*hd)
    return W[f'W_q.{l}'][s]*w, W[f'W_k.{l}'][s]*w, W[f'W_v.{l}'][s]*w, W[f'W_o.{l}'][:,s]
def orth(M,r):
    u,s,vt=np.linalg.svd(M,full_matrices=False); return u[:,:r]
q,k,v,o=hw(0,2); A=q.T@k; u,s,vt=np.linalg.svd(A)
P=orth(np.hstack([u[:,:18],vt[:18].T]),20)
ov2=o@v; uo,so,vo=np.linalg.svd(ov2)
T=vo[:24].T; O2=uo[:,:24]
Obs={i:np.linalg.svd(hw(0,i)[3]@hw(0,i)[2])[0][:,:3] for i in [0,1,3,4,5]}
Ob=orth(np.hstack([Obs[i] for i in [0,1,4,5]]),12)
M0=orth(W['W_proj.0'],60)
subs=dict(P=P,T=T,O2=O2,Ob=Ob,M0=M0)
for L in [1,2]:
  for j in range(6):
    q,k,v,o=hw(L,j); QK=q.T@k
    tot=np.linalg.norm(QK)
    print(L,j,'key=Ob:',' '.join('%s %.2f'%(nm,np.linalg.norm(S.T@QK@Ob)/tot) for nm,S in subs.items()),
          '| key=T:',' '.join('%s %.2f'%(nm,np.linalg.norm(S.T@QK@T)/tot) for nm,S in subs.items()),
          '| key=O2:',' '.join('%s %.2f'%(nm,np.linalg.norm(S.T@QK@O2)/tot) for nm,S in subs.items()))
```

````