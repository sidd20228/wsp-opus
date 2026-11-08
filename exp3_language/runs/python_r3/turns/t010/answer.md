```python
import numpy as np
def comp(A,B): return np.linalg.norm(A@B)/(np.linalg.norm(A)*np.linalg.norm(B))
rng=np.random.default_rng(0)
P0=W['W_proj.0']
w1=W['ln1_w.0']; H=w1>0.45
E_H=np.eye(384)[:,H]; E_T=np.eye(384)[:,~H]
for l in [1,2,3]:
    wl=W[f'ln1_w.{l}']
    for j in range(6):
        s=slice(64*j,64*j+64)
        row=[]
        for n in ['W_q','W_k','W_v']:
            A=W[f'{n}.{l}'][s]*wl
            cH=(A[:,H]**2).sum()/(A**2).sum()
            row.append(f"{n[-1]}: mlp0 {comp(A,P0):.2f} Hfrac {cH:.2f}")
        print(f'L{l}H{j}', ' | '.join(row))
R=rng.normal(size=(64,384)); print('random baseline mlp0', comp(R,P0))
print('ln1_w.1 on H', W['ln1_w.1'][H].mean(), 'off', W['ln1_w.1'][~H].mean())
print('ln1_w.2 on H', W['ln1_w.2'][H].mean(), 'off', W['ln1_w.2'][~H].mean())
```