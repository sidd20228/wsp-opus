```python
import numpy as np
def rows(m,j,l,fold=True): 
    s=slice(j*64,(j+1)*64); return W[f'W_{m}.{l}'][s]*(W[f'ln1_w.{l}'] if fold else 1)
def OV(l,j):
    s=slice(j*64,(j+1)*64); return W[f'W_o.{l}'][:,s]@rows('v',j,l)
A=rows('q',1,3).T@rows('k',1,3)
M=A@OV(0,2)
ev=np.linalg.eigvals(M); o=np.argsort(-np.abs(ev)); ev=ev[o]
print(' '.join(f'{abs(e):.3f}@{np.degrees(np.angle(e)):.0f}' for e in ev[:40]))
print('real parts sorted',np.sort(ev.real)[::-1][:40].round(3))
sv=np.linalg.svd(M,compute_uv=False); print('sv',sv[:40].round(3))
```