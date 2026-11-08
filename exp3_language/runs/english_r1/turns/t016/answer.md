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
q,k,v,o=hw(0,2); A=q.T@k; u,s,vt=np.linalg.svd(A)
P=orth(np.hstack([u[:,:18],vt[:18].T]),20)
ov2=o@v; uo,so,vo=np.linalg.svd(ov2)
T=vo[:24].T
# orthogonalize T against P to get pure token part, and P against T
Tp=orth(T-P@(P.T@T),24)
# total active input space of layer0
w=W['ln1_w.0']
R=np.vstack([W['W_q.0']*w,W['W_k.0']*w,W['W_v.0']*w])
sR=np.linalg.svd(R,compute_uv=False); print('stacked L0 attn read sv',(sR[:80]).round(1))
for j in range(6):
    q,k,v,o=hw(0,j); QK=q.T@k
    u,s,vt=np.linalg.svd(QK)
    for r in range(2):
        uq=u[:,r]; vk=vt[r]
        print(j,r,'s*48=%.1f'%(s[r]*48),'q: P %.2f T %.2f rest %.2f'%(np.sum((P.T@uq)**2),np.sum((Tp.T@uq)**2),1-np.sum((P.T@uq)**2)-np.sum((Tp.T@uq)**2)),
              '| k: P %.2f T %.2f rest %.2f'%(np.sum((P.T@vk)**2),np.sum((Tp.T@vk)**2),1-np.sum((P.T@vk)**2)-np.sum((Tp.T@vk)**2)))
```

````