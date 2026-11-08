```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
w=W['ln1_w.0']
U,S,Vt=np.linalg.svd(W['W_v.0'][256:320]*w,full_matrices=False); D=Vt[:3].T
def planes(j, n=6):
    s=slice(64*j,64*j+64)
    Q=W['W_q.0'][s]*w; K=W['W_k.0'][s]*w
    M=Q.T@K
    ev,V=np.linalg.eig(M)
    o=np.argsort(-np.abs(ev))
    res=[]
    for i in o:
        e=ev[i]
        if e.imag< -1e-8: continue
        v=V[:,i]
        B=np.stack([v.real,v.imag],1) if abs(e.imag)>1e-8 else v.real[:,None]
        Qb,_=np.linalg.qr(B); res.append((e,Qb))
        if len(res)>=n: break
    return res
P1=planes(1,8)
for j in [0,2,3]:
    for e,B in planes(j,4):
        ov=[np.linalg.norm(B1.T@B) for e1,B1 in P1]
        print(f'L0H{j} {abs(e):.2f}@{np.degrees(np.angle(e)):.0f}', 'D-overlap', np.linalg.norm(D.T@B,axis=0).round(2), 'L0H1 planes', ' '.join(f"{abs(e1):.1f}@{np.degrees(np.angle(e1)):.0f}:{x:.2f}" for (e1,_),x in zip(P1,ov)))
```