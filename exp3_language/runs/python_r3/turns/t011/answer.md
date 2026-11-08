```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
def hq(l,j):
    s=slice(64*j,64*j+64); w=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][s]*w, W[f'W_k.{l}'][s]*w, W[f'W_v.{l}'][s]*w, W[f'W_o.{l}'][:,s]
def stat(N):
    ev=np.linalg.eigvals(N)
    a=np.abs(ev); 
    o=np.argsort(-a)
    top=ev[o[:4]]
    return ev.real.sum()/a.sum(), a.sum(), top
srcs=[(0,i) for i in range(6)]
for L in [1,2,3]:
    for h in range(6):
        q,k,v,o=hq(L,h)
        QK=q.T@k
        out=[]
        for (ls,i) in srcs+[(1,i) for i in range(6) if L>1]:
            qs,ks,vs,os_=hq(ls,i)
            OV=os_@vs
            r,mag,top=stat(QK@OV)
            out.append(f"{ls}.{i}:{r:+.2f}/{mag:.1f}")
        print(f"L{L}H{h}", ' '.join(out))
```