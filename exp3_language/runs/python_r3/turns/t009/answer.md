```python
import numpy as np
w1=W['ln1_w.0']
H = w1>0.45
print('num high dims', H.sum())
for j in range(6):
    s=slice(64*j,64*j+64)
    out=[]
    for n in ['W_q','W_k','W_v']:
        M=W[f'{n}.0'][s]*w1
        cn=(M**2).sum(0)
        out.append(f"{n[-1]}: {cn[H].sum()/cn.sum():.2f} (tot {np.sqrt(cn.sum()):.2f})")
    print(f'L0H{j}', ' | '.join(out))
w2=W['ln2_w.0']
M=W['W_fc.0']*w2
cn=(M**2).sum(0); print('fc0', cn[H].sum()/cn.sum(), 'w2 mean on H', w2[H].mean(), 'off', w2[~H].mean())
# raw, unscaled column norms
for n in ['W_q','W_k','W_v']:
    M=W[f'{n}.0']; cn=(M**2).sum(0); print(n,'raw frac H', round(float(cn[H].sum()/cn.sum()),3), 'H frac dims', H.mean().round(3))
```