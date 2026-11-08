```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
w0=W['ln1_w.0']; H=w0>0.45
rows=np.concatenate([W['W_v.0'][64*j:64*j+64] for j in [1,2,3,5]],0)
U_,S_,Vt_=np.linalg.svd(rows,full_matrices=False)
T=Vt_[:27].T
us=[]
for L in [1,2,3]:
    for h in range(6):
        s=slice(64*h,64*h+64); wl=W[f'ln1_w.{L}']
        k=W[f'W_k.{L}'][s]*wl
        qb=W[f'W_q.{L}'][s]@W[f'ln1_b.{L}']
        u=T.T@(k.T@qb)
        us.append((f'L{L}H{h}',u))
names=[n for n,_ in us]
Ucat=np.array([u/np.linalg.norm(u) for _,u in us])
C=Ucat@Ucat.T
print('      '+' '.join(f'{n:>5}' for n in names))
for i,n in enumerate(names):
    print(f'{n:>5} '+' '.join(f'{x:5.2f}' for x in C[i]), f' |u|={np.linalg.norm(us[i][1]):.2f}')
```