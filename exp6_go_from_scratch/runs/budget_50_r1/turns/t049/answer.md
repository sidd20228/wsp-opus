73% vs 83% for v6 — likely worse (about 2 SE). Let me check what the losses look like before deciding.

```python
import json, numpy as np
from collections import Counter
G=json.load(open('/work/pass_9/games.json')); B=np.load('/work/pass_9/boards.npy')
print(Counter((g['opponent'],g['end'],g['result']) for g in G))
for gi,g in enumerate(G):
    if g['result']==-1:
        n=len(g['moves'])
        s=f"{gi} {g['opponent']} {g['net_color']} {g['end']} margin {g['margin']} len {n}"
        if g['end']=='repetition':
            last=B[gi,n-1]; k0=[k for k in range(n-1) if np.array_equal(B[gi,k],last)][-1]
            s+=' cycle '+str([(m['by'][0], divmod(m['move'],9) if m['move']<81 else 'P', round(m['top5'][0][1],2) if 'top5' in m else '') for m in g['moves'][k0+1:n]][:8])
        print(s)
```