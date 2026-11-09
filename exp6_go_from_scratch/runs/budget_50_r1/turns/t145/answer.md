Ties v29 on wins (125) with a higher margin. Let me check the paired discordance before deciding.

```python
import json
from collections import Counter
A=json.load(open('/work/pass_33/games.json')); Bv=json.load(open('/work/pass_38/games.json'))
print(Counter((a['result'],b['result']) for a,b in zip(A,Bv)))
import numpy as np
ma=np.array([a['margin'] for a in A]); mb=np.array([b['margin'] for b in Bv])
d=mb-ma; print('paired margin diff mean %.2f se %.2f'%(d.mean(), d.std()/np.sqrt(len(d))))
```