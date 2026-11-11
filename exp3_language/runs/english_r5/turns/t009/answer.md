```python
import numpy as np
def rows(m,j,l=0): 
    s=slice(j*64,(j+1)*64); return W[f'W_{m}.{l}'][s]*W[f'ln1_w.{l}']
for (l,j) in [(0,2),(0,3)]:
    Wq=rows('q',j,l);Wk=rows('k',j,l)
    ev=np.linalg.eigvals(Wk@Wq.T)
    o=np.argsort(-np.abs(ev))
    print(f'L{l}H{j}')
    for e in ev[o][:24]:
        print(f'  |{abs(e):.2f}| phase {np.degrees(np.angle(e)):7.1f}')
```