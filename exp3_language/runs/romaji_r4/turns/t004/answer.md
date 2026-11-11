```python
import numpy as np
fro=np.linalg.norm
H=6;hd=64
def head(l,j):
    g=W[f'ln1_w.{l}'];s=slice(j*hd,(j+1)*hd)
    return W[f'W_q.{l}'][s]*g, W[f'W_k.{l}'][s]*g, W[f'W_v.{l}'][s]*g, W[f'W_o.{l}'][:,s]
for (l,j) in [(0,0),(0,3),(0,1),(3,3),(3,4),(1,0),(2,0)]:
    Q,K,V,O=head(l,j);M=Q.T@K
    S=(M+M.T)/2;A=(M-M.T)/2
    ev=np.linalg.eigvals(M);ev=ev[np.argsort(-np.abs(ev))]
    es=np.linalg.eigvalsh(S)
    print(f'L{l}H{j} |S| %.2f |A| %.2f  S eig top %s bottom %s'%(fro(S),fro(A),np.round(es[-4:],2),np.round(es[:4],2)))
    print('   eig(M):',' '.join('%.2f@%d'%(abs(e),np.degrees(np.angle(e))) for e in ev[:16]))
```