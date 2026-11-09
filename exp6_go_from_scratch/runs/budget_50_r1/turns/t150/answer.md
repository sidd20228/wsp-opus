127/128 with α=1.0. Paired check against v29:

```python
import json, numpy as np
from collections import Counter
A=json.load(open('/work/pass_33/games.json')); Bv=json.load(open('/work/pass_40/games.json'))
print(Counter((a['result'],b['result']) for a,b in zip(A,Bv)))
ma=np.array([a['margin'] for a in A]); mb=np.array([b['margin'] for b in Bv])
d=mb-ma; print('paired margin diff mean %.2f se %.2f'%(d.mean(), d.std()/np.sqrt(len(d))))
print([(i,b['net_color'],b['margin'],b['end']) for i,b in enumerate(Bv) if b['result']==-1])
```