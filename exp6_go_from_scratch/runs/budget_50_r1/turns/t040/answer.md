Repetitions went up (17) and passes per game are high again (the sharper positional scale pushed moves below the pass logit). Let me check the cycles.

```python
import json, numpy as np
from collections import Counter
G=json.load(open('/work/pass_7/games.json')); B=np.load('/work/pass_7/boards.npy')
print(Counter((g['opponent'],g['end'],g['result']) for g in G))
cl=Counter()
for gi,g in enumerate(G):
    if g['end']!='repetition': continue
    n=len(g['moves']); last=B[gi,n-1]
    k0=[k for k in range(n-1) if np.array_equal(B[gi,k],last)][-1]
    seq=[(m['by'][0], divmod(m['move'],9) if m['move']<81 else 'P') for m in g['moves'][k0+1:n]]
    cl[n-1-k0]+=1
    print(gi,g['margin'],'len',n-1-k0, seq[:10])
print(cl)
fp=[]
for g in G:
    for m in g['moves']:
        if m['by']=='net' and m['move']==81: fp.append(m['ply']); break
print(sorted(fp))
```