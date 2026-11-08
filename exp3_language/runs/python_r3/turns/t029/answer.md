```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=250)
w0=W['ln1_w.0']; H=w0>0.45
for j in range(6):
    S=np.linalg.svd(W['W_v.0'][64*j:64*j+64][:,~H],compute_uv=False)
    print(f'V{j} nonH', S[:34], ' ratios s22..s29:', (S[21:29]/S[22:30]).round(2))
for nm in ['W_q.0','W_k.0']:
    S=np.linalg.svd(W[nm][:,~H],compute_uv=False); print(nm,'nonH all heads', S[:34])
S=np.linalg.svd(W['W_fc.0'][:,~H],compute_uv=False); print('fc0 nonH', (S[:40]).round(2))
# random baseline
R=np.random.default_rng(1).normal(0,0.02,size=(64,int((~H).sum())))
print('random 64x314 sv', np.linalg.svd(R,compute_uv=False)[:3])
```