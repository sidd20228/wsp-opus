```python
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=200)
w = W['ln1_w.0']
print(np.histogram(w, bins=20))
print(np.sort(w)[:20], np.sort(w)[-20:])
w2 = W['ln2_w.0']
print(np.histogram(w2, bins=20))
print(np.corrcoef(w, w2))
for l in range(4):
    for name in ['W_q','W_k','W_v']:
        M = W[f'{name}.{l}']
        print(l, name, [round(float(np.linalg.norm(M[64*j:64*j+64])),2) for j in range(6)])
    M = W[f'W_o.{l}']
    print(l, 'W_o', [round(float(np.linalg.norm(M[:,64*j:64*j+64])),2) for j in range(6)])
```