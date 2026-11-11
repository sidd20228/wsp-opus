```python
import numpy as np
def rows(m,j,l,fold=True): 
    s=slice(j*64,(j+1)*64); return W[f'W_{m}.{l}'][s]*(W[f'ln1_w.{l}'] if fold else 1)
def OV(l,j):
    s=slice(j*64,(j+1)*64); return W[f'W_o.{l}'][:,s]@rows('v',j,l)
fc=W['W_fc.0']; alive=np.linalg.norm(fc,axis=1)>0.6
F=fc[alive]*W['ln2_w.0']
print('alive',alive.sum())
print('alive fc sv',np.linalg.svd(F,compute_uv=False)[:70].round(2))
# remove attention-output spaces (all L0 heads Wo column spaces) 
Wo=W['W_o.0']; uo,so,_=np.linalg.svd(Wo); 
print('Wo0 sv', so[:70].round(2))
```