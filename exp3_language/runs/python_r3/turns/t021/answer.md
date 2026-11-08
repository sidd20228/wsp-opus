```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
w=W['ln1_w.0']; H=w>0.45
rows=np.concatenate([W['W_v.0'][64*j:64*j+64] for j in [1,2,3,5]],0)
T=np.linalg.svd(rows,full_matrices=False)[2][:27].T
for j in [1,5,2,3,0]:
    s=slice(64*j,64*j+64)
    Q=W['W_q.0'][s]*w; K=W['W_k.0'][s]*w
    M=Q.T@K
    ev,V=np.linalg.eig(M)
    o=np.argsort(-np.abs(ev))
    c=0
    for i in o:
        e=ev[i]
        if e.imag<-1e-8: continue
        v=V[:,i]; B=np.stack([v.real,v.imag],1) if abs(e.imag)>1e-8 else v.real[:,None]
        Qb,_=np.linalg.qr(B)
        hf=(Qb[H]**2).sum()/Qb.shape[1]; tf=(np.linalg.norm(T.T@Qb)**2)/Qb.shape[1]
        print(f'L0H{j} {abs(e):.2f}@{np.degrees(np.angle(e)):.0f}  Hfrac {hf:.2f}  Tfrac {tf:.2f}')
        c+=1
        if c>=8: break
```