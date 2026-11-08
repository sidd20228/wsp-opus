Better: 67% strong, 94% weak. Still 8+1+2 repetition endings. Let me look at those cycles and the remaining losses.

```python
import json, numpy as np
from collections import Counter
G=json.load(open('/work/pass_5/games.json')); B=np.load('/work/pass_5/boards.npy')
def show(b):
    ch={1:'X',-1:'O',0:'.'}
    return [''.join(ch[int(b[r*9+c])] for c in range(9)) for r in range(9)]
print(Counter((g['opponent'],g['end'],g['result']) for g in G))
for gi,g in enumerate(G):
    if g['end']!='repetition': continue
    n=len(g['moves']); last=B[gi,n-1]
    k0=[k for k in range(n-1) if np.array_equal(B[gi,k],last)][-1]
    seq=[(m['by'][0], divmod(m['move'],9) if m['move']<81 else 'P', [(divmod(a,9) if a<81 else 'P',round(p,2)) for a,p in m.get('top5',[])[:2]]) for m in g['moves'][k0+1:n]]
    print(gi,g['opponent'],g['margin'],'cycle len',n-1-k0)
    for s in seq[:6]: print('    ',s)
```