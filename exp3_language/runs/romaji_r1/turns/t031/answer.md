```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=200)
for l in range(4):
    Wfc = W[f'W_fc.{l}']*W[f'ln2_w.{l}']
    n_in = np.linalg.norm(Wfc, axis=1)
    n_out = np.linalg.norm(W[f'W_proj.{l}'], axis=0)
    print(l, 'in-norm pct', np.percentile(n_in,[5,25,50,75,95]), ' out-norm pct', np.percentile(n_out,[5,25,50,75,95]))
    print('   hist in', np.histogram(n_in, bins=8)[0], np.round(np.histogram(n_in, bins=8)[1],2))
    print('   #neurons with out-norm > 2x median-of-low', (n_out > 0.5).sum(), (n_out>1).sum())
```