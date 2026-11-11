```python
import numpy as np
def rows(m,j,l): 
    s=slice(j*64,(j+1)*64); return W[f'W_{m}.{l}'][s]*W[f'ln1_w.{l}']
def OV(l,j):
    s=slice(j*64,(j+1)*64); return W[f'W_o.{l}'][:,s]@rows('v',j,l)
def QK(l,j): return rows('q',j,l).T@rows('k',j,l)
def pos(M):
    ev=np.linalg.eigvals(M); return ev.real.sum()/np.abs(ev).sum()
def comp(A,B): return np.linalg.norm(A@B)/(np.linalg.norm(A)*np.linalg.norm(B))
O2=OV(0,2)
for l in [1,2,3]:
    for j in range(6):
        A=QK(l,j)
        qc=O2.T@A    # query reads L0H2 output (prev token at i); key reads raw
        kc=A@O2      # key reads L0H2 output at j
        qk=O2.T@A@O2 # both prev tokens
        print(f'L{l}H{j} Qcomp(prev_i vs tok_j): pos={pos(qc):+.2f} comp={comp(O2.T,A):.3f} | Kcomp(tok_i vs prev_j): pos={pos(kc):+.2f} comp={comp(A,O2):.3f} | prev-prev pos={pos(qk):+.2f}')
```