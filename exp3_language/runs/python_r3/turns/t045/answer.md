```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
w0=W['ln1_w.0']
rows=np.concatenate([W['W_v.0'][64*j:64*j+64] for j in [1,2,3,5]],0)
T=np.linalg.svd(rows,full_matrices=False)[2][:27].T
def hq(l,j):
    s=slice(64*j,64*j+64); wl=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][s]*wl, W[f'W_k.{l}'][s]*wl, W[f'W_v.{l}'][s]*wl, W[f'W_o.{l}'][:,s]
def tr(A): return np.trace(A)/np.linalg.norm(A,'nuc'), np.linalg.norm(A,'nuc')
OV0={i:hq(0,i)[3]@hq(0,i)[2] for i in range(6)}
OV1={i:hq(1,i)[3]@hq(1,i)[2] for i in range(6)}
OV2={i:hq(2,i)[3]@hq(2,i)[2] for i in range(6)}
for (L,h) in [(2,0),(2,3),(2,4),(2,5),(3,0),(3,2),(3,3),(3,4)]:
    q,k,v,o=hq(L,h); M=q.T@k
    out=[]
    for nm,src in [('L1',OV1),('L2',OV2)]:
        if nm=='L2' and L==2: continue
        for i in range(6):
            r,m=tr(T.T@M@src[i]@OV0[5]@T)   # key holds L-head-moved prev-token info
            r2,m2=tr(T.T@M@src[i]@T)
            out.append(f"{nm}H{i}:T{r2:+.2f}/{m2:.1f} p{r:+.2f}")
    print(f'L{L}H{h}', ' '.join(out))
```