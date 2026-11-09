```python
import numpy as np
hd=64
def OVs(l):
    lw=W[f'ln1_w.{l}']
    return [W[f'W_o.{l}'][:,j*hd:(j+1)*hd]@(W[f'W_v.{l}'][j*hd:(j+1)*hd]*lw) for j in range(6)]
OV=[OVs(l) for l in range(4)]
def ratio(A,B):
    return (np.linalg.norm(A@B)**2/np.linalg.norm(B)**2)/(np.linalg.norm(A)**2/384)
for l in [1,2,3]:
    lw=W[f'ln1_w.{l}']
    print('=== layer',l,' K-read / Q-read / V-read ratios of writers: L0H0..5 | L1H0..5 | L2H0..5 | MLP0 MLP1 MLP2')
    for j in range(6):
        s=slice(j*hd,(j+1)*hd)
        Wk=W[f'W_k.{l}'][s]*lw; Wq=W[f'W_q.{l}'][s]*lw; Wv=W[f'W_v.{l}'][s]*lw
        for nm,A in [('K',Wk),('Q',Wq),('V',Wv)]:
            r=[]
            for l1 in range(l):
                r+= [ratio(A,OV[l1][h]) for h in range(6)]
            r+= [ratio(A,W[f'W_proj.{l1}']) for l1 in range(l)]
            print(j,nm,' '.join('%5.1f'%x for x in r))
```