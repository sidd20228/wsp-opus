```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
w1=W['ln1_w.0']
tops={}
for j in range(6):
    s=slice(64*j,64*j+64)
    for n in ['W_q','W_k','W_v']:
        M=W[f'{n}.0'][s]*w1
        U,S,Vt=np.linalg.svd(M,full_matrices=False)
        for r in range(3):
            tops[f'{n[-1]}{j}_{r}']=(Vt[r],S[r])
keys=[k for k in tops if tops[k][1]>0.9]
print(keys)
C=np.array([[abs(tops[a][0]@tops[b][0]) for b in keys] for a in keys])
print('       '+' '.join(f'{k:>6}' for k in keys))
for i,a in enumerate(keys):
    print(f'{a:>6} '+' '.join(f'{x:6.2f}' for x in C[i]))
```