```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
w=W['ln1_w.0']
s4=slice(256,320)
U,S,Vt=np.linalg.svd(W['W_v.0'][s4]*w,full_matrices=False)
D=Vt[:3].T
def planes(j, n=10):
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
for j in [1,5]:
    for e,B in planes(j):
        print(f'L0H{j} {abs(e):.2f}@{np.degrees(np.angle(e)):.0f}', 'overlap with D1,D2,D3:', np.abs(D.T@B).round(2).tolist())
```