```python
import numpy as np
def rows(m,j,l): 
    s=slice(j*64,(j+1)*64); return W[f'W_{m}.{l}'][s]*W[f'ln1_w.{l}']
def OV(l,j):
    s=slice(j*64,(j+1)*64); return W[f'W_o.{l}'][:,s]@rows('v',j,l)
def QK(l,j): return rows('q',j,l).T@rows('k',j,l)
def pos(M):
    ev=np.linalg.eigvals(M); return ev.real.sum()/np.abs(ev).sum(), np.abs(ev).max()
res=[]
for l in [1,2,3]:
    for j in range(6):
        r=[]
        for src in [2,3]:
            M=QK(l,j)@OV(0,src)
            p,mx=pos(M)
            r.append(f'L0H{src}->K: pos={p:+.2f}')
        # also direct "same token" matching: QK positivity itself
        p,_=pos(QK(l,j))
        print(f'L{l}H{j}',' '.join(r),f'| QK self-pos {p:+.2f}')
```