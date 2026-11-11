```python
import numpy as np
def rows(m,j,l): 
    s=slice(j*64,(j+1)*64); return W[f'W_{m}.{l}'][s]*W[f'ln1_w.{l}']
def OV(l,j):
    s=slice(j*64,(j+1)*64); return W[f'W_o.{l}'][:,s]@rows('v',j,l)
def comp(A,B): return np.linalg.norm(A@B)/(np.linalg.norm(A)*np.linalg.norm(B))
rng=np.random.default_rng(1)
Rnd=rng.standard_normal((384,1536))
for (l,j) in [(1,0),(1,2),(2,0),(2,2),(3,0),(3,1),(3,2),(3,5)]:
    Wq=rows('q',j,l);Wk=rows('k',j,l);Wv=rows('v',j,l)
    out=[]
    for src in range(l):
        P=W[f'W_proj.{src}']
        out.append(f'mlp{src}: q{comp(Wq,P):.3f} k{comp(Wk,P):.3f} v{comp(Wv,P):.3f}')
    out.append(f'rand: q{comp(Wq,Rnd):.3f} k{comp(Wk,Rnd):.3f} v{comp(Wv,Rnd):.3f}')
    print(f'L{l}H{j}',' | '.join(out))
```