```python
import numpy as np
def rows(m,j,l): 
    s=slice(j*64,(j+1)*64); return W[f'W_{m}.{l}'][s]*W[f'ln1_w.{l}']
for l in [1,2,3]:
    for j in range(6):
        Wq=rows('q',j,l);Wk=rows('k',j,l)
        ev=np.linalg.eigvals(Wk@Wq.T)
        o=np.argsort(-np.abs(ev))[:14]
        print(f'L{l}H{j}',' '.join(f'{abs(e):.1f}@{np.degrees(np.angle(e)):.0f}' for e in ev[o]))
```