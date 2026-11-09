The losses come from small territorial deficits; nothing obviously broken. Before tuning further, let me check how often big own groups die (life-and-death failures), since those are the most costly events.

```python
import json, numpy as np
from collections import Counter
G=json.load(open('/work/pass_17/games.json')); B=np.load('/work/pass_17/boards.npy')
big=Counter()
for gi,g in enumerate(G):
    mine=1 if g['net_color']=='black' else -1
    n=len(g['moves'])
    for k in range(1,n):
        if g['moves'][k]['by']!='opponent': continue
        l=int(((B[gi,k-1]==mine)&(B[gi,k]==0)).sum())
        if l>=5: big[(gi,g['result'])]+=l
print(sorted(big.items(), key=lambda x:-x[1])[:20])
print('games with >=5-stone loss events:',len(big), ' among losses:',sum(1 for (gi,r) in big if r==-1))
```