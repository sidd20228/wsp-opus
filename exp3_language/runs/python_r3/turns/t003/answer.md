```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
w1 = W['ln1_w.0']; b1=W['ln1_b.0']
idx = np.argsort(-w1)
print("top ln1_w dims", idx[:40])
print(w1[idx[:40]])
for name in ['W_q','W_k','W_v']:
    M = W[f'{name}.0']
    for j in range(6):
        cn = np.linalg.norm(M[64*j:64*j+64], axis=0)  # per input dim
        eff = cn*np.abs(w1)
        # how concentrated
        s = np.sort(eff)[::-1]
        print(name, j, 'top10 eff share', round(float((s[:10]**2).sum()/(s**2).sum()),3), 'top40', round(float((s[:40]**2).sum()/(s**2).sum()),3), 'corr colnorm vs w1', round(float(np.corrcoef(cn,w1)[0,1]),3))
```