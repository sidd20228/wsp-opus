<invoke name="python">
```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=200)
H=6; hd=64
def hw(l,j):
    w=W[f'ln1_w.{l}']; s=slice(j*hd,(j+1)*hd)
    return W[f'W_q.{l}'][s]*w, W[f'W_k.{l}'][s]*w, W[f'W_v.{l}'][s]*w, W[f'W_o.{l}'][:,s]
Wq,Wk,Wv,Wo=hw(0,2)
A=Wq.T@Wk
u,s,vt=np.linalg.svd(A)
P=np.linalg.qr(np.hstack([u[:,:18],vt[:18].T]))[0]
# reduce to 20-dim
up,sp,_=np.linalg.svd(np.hstack([u[:,:18],vt[:18].T]),full_matrices=False)
P=up[:,:20]
ov=Wo@Wv
uo,so,vo=np.linalg.svd(ov)
T=vo[:24].T
print('overlap P vs T (OV input space of H2):', np.linalg.svd(P.T@T,compute_uv=False).round(2))
for i in range(6):
    q,k,v,o=hw(0,i)
    OV=o@v
    _,_,vv=np.linalg.svd(OV); 
    QK=q.T@k; uq,sq,vq=np.linalg.svd(QK)
    f=lambda x,S: float(np.linalg.norm(S.T@x)**2)
    print(i,'OV in-dirs: frac in P',[round(f(vv[r],P),2) for r in range(3)],'frac in T',[round(f(vv[r],T),2) for r in range(3)],
          '| QK q-dir1 P %.2f T %.2f ; k-dir1 P %.2f T %.2f'%(f(uq[:,0],P),f(uq[:,0],T),f(vq[0],P),f(vq[0],T)))
```

````