```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
w=W['ln1_w.0']
def planes(j, n=8):
    s=slice(64*j,64*j+64)
    Q=W['W_q.0'][s]*w; K=W['W_k.0'][s]*w
    M=Q.T@K
    ev,V=np.linalg.eig(M)
    o=np.argsort(-np.abs(ev))
    res=[]
    seen=0
    for i in o:
        e=ev[i]
        if e.imag< -1e-8: continue
        v=V[:,i]
        if abs(e.imag)>1e-8:
            B=np.stack([v.real,v.imag],1)
        else:
            B=v.real[:,None]
        Qb,_=np.linalg.qr(B)
        res.append((e,Qb))
        if len(res)>=n: break
    return res
P1=planes(1,8); P5=planes(5,10)
print('rows L0H1 planes, cols L0H5 planes: sqrt(sum cos^2)')
print('            '+' '.join(f"{abs(e):.1f}@{np.degrees(np.angle(e)):.0f}".rjust(9) for e,_ in P5))
for e1,B1 in P1:
    row=[np.linalg.norm(B1.T@B5) for e5,B5 in P5]
    print(f"{abs(e1):.1f}@{np.degrees(np.angle(e1)):.0f}".rjust(10), ' '.join(f"{x:9.2f}" for x in row))
```