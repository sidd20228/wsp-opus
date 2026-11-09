```python
import numpy as np
np.set_printoptions(linewidth=220)
hd=64
def topRows(M,r):
    u,s,vt=np.linalg.svd(M,full_matrices=False); return vt[:r]
sl=lambda j: slice(j*hd,(j+1)*hd)
def rd(nm,l,j): return W[f'{nm}.{l}'][sl(j)]*W[f'ln1_w.{l}']
U=np.vstack([topRows(rd('W_q',0,3),30),topRows(rd('W_k',0,3),30)]); u,s,vt=np.linalg.svd(U); P=vt[:30]
OVs={j:W['W_o.0'][:,sl(j)]@rd('W_v',0,j) for j in range(6)}
outsp={}
for j in range(6):
    uu,ss,vv=np.linalg.svd(OVs[j]); outsp[j]=uu[:,:(3 if j in (1,2,4,5) else 20)].T
    print('L0H',j,'OV top sv',np.round(ss[:4],2))
for L in [1,2,3]:
    b=W[f'ln1_b.{L}']
    for h in range(6):
        qb=W[f'W_q.{L}'][sl(h)]@b
        k=rd('W_k',L,h)
        R=k.T@qb; R=R/np.linalg.norm(R)
        parts={'P':np.linalg.norm(P@R)}
        for j in range(6): parts[f'o{j}']=np.linalg.norm(outsp[j]@R)
        print(f'L{L}H{h}',' '.join(f'{a}:{c:.2f}' for a,c in parts.items()))
# relation among outputs of L0H1,2,4,5
for a in [1,2,4,5]:
    print(a,[np.round(np.linalg.svd(outsp[a]@outsp[c].T,compute_uv=False),2) for c in [1,2,4,5]])
```