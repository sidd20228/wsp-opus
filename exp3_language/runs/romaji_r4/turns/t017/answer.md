```python
import numpy as np
fro=np.linalg.norm
H=6;hd=64
def head(l,j):
    g=W[f'ln1_w.{l}'];s=slice(j*hd,(j+1)*hd)
    return W[f'W_q.{l}'][s]*g, W[f'W_k.{l}'][s]*g, W[f'W_v.{l}'][s]*g, W[f'W_o.{l}'][:,s]
def top(M,k): return np.linalg.svd(M,full_matrices=False)[0][:,:k]
srcs={'E':np.linalg.svd(W['W_fc.0'],full_matrices=False)[2][:40].T}
for l in range(3):
    for j in range(H): srcs[f'A{l}{j}']=top(head(l,j)[3]@head(l,j)[2],8)
    srcs[f'M{l}']=top(W[f'W_proj.{l}'],32)
for (l,j) in [(3,4),(3,3),(3,0),(2,4),(1,0)]:
    Q,K,V,O=head(l,j);OV=O@V
    uo,so,vo=np.linalg.svd(OV)
    for nm,vec in [('Vtop1',vo[0]),('Vtop2',vo[1]),('Vtop3',vo[2])]:
        sc=sorted([(k,fro(P.T@vec)**2/(P.shape[1]/384)) for k,P in srcs.items() if k=='E' or int(k[1])<l],key=lambda t:-t[1])[:4]
        print(f'L{l}H{j} {nm} (sv {so[["Vtop1","Vtop2","Vtop3"].index(nm)]:.1f}):',' '.join('%s x%.1f'%t for t in sc))
    sc=sorted([(k,fro(V@P)**2/fro(V)**2/(P.shape[1]/384)) for k,P in srcs.items() if k=='E' or int(k[1])<l],key=lambda t:-t[1])[:6]
    print(f'L{l}H{j} whole V:',' '.join('%s x%.1f'%t for t in sc))
```