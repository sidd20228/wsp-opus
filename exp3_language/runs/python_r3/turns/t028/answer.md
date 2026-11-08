```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
w0=W['ln1_w.0']
rows=np.concatenate([W['W_v.0'][64*j:64*j+64] for j in [1,2,3,5]],0)
T=np.linalg.svd(rows,full_matrices=False)[2][:27].T
def hq(l,j):
    s=slice(64*j,64*j+64); wl=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][s]*wl, W[f'W_k.{l}'][s]*wl, W[f'W_v.{l}'][s]*wl, W[f'W_o.{l}'][:,s]
OV={i:hq(0,i)[3]@hq(0,i)[2] for i in range(6)}
srcs={'T':np.eye(384),'OV1':OV[1],'OV5':OV[5],'OV2':OV[2],'OV3':OV[3]}
def tr(A):
    return np.trace(A)/np.linalg.norm(A,'nuc'), np.linalg.norm(A,'nuc')
for (L,h) in [(2,1),(3,1),(3,5),(3,0),(2,5),(3,4),(1,1),(1,5),(2,2)]:
    q,k,v,o=hq(L,h); M=q.T@k
    print(f'L{L}H{h}')
    for qs in ['T','OV1','OV5']:
        row=[]
        for ks in ['T','OV1','OV5','OV2','OV3']:
            A=T.T@srcs[qs].T@M@srcs[ks]@T
            r,m=tr(A); row.append(f"k:{ks} {r:+.2f}/{m:.2f}")
        print(f'   q:{qs:4}', ' | '.join(row))
```