α=0.7–0.9 pooled: 373/384 = 97.1% vs 94.5% at α=0.5. Let me check the current loss composition.

```python
import json, numpy as np
from collections import Counter
for pk in [25,26,27]:
    G=json.load(open('/work/pass_%d/games.json'%pk)); B=np.load('/work/pass_%d/boards.npy'%pk)
    for gi,g in enumerate(G):
        if g['result']==-1:
            n=len(g['moves']); s=f"pass{pk} g{gi} {g['net_color']} {g['end']} {g['margin']}"
            if g['end']=='repetition':
                last=B[gi,n-1]; k0=[k for k in range(n-1) if np.array_equal(B[gi,k],last)][-1]
                s+=' cyc'+str([(m['by'][0], divmod(m['move'],9) if m['move']<81 else 'P') for m in g['moves'][k0+1:n]])
            print(s)
```