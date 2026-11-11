```python
import numpy as np
d='/work/pass_1/'
L=np.load(d+'logits.npy'); Pr=np.load(d+'probs.npy'); lg=np.load(d+'legal.npy')
def show(i):
    l=L[i]
    print('pos',i,'pass',round(float(l[81]),2))
    for r in range(9):
        print(' '.join(f'{l[r*9+c]:6.1f}' for c in range(9)))
for i in [0,4,7]: show(i)
```