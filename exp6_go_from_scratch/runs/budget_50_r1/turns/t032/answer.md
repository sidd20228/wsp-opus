Real single-stone captures dominate these cases (670 vs 69 false), so I can't just penalize SA-captures. Let me look at the non-repetition losses in this pass before deciding what to change.

```python
import json, numpy as np
from collections import Counter
G=json.load(open('/work/pass_5/games.json')); B=np.load('/work/pass_5/boards.npy')
def show(b):
    ch={1:'X',-1:'O',0:'.'}
    return [''.join(ch[int(b[r*9+c])] for c in range(9)) for r in range(9)]
L=[(gi,g) for gi,g in enumerate(G) if g['result']==-1 and g['end']!='repetition']
print([(gi,g['opponent'],g['net_color'],g['margin'],len(g['moves'])) for gi,g in L])
for gi,g in L[:4]:
    n=len(g['moves'])
    print('game',gi,g['net_color'],g['margin'])
    rows=[show(B[gi,j]) for j in (19,39,59,n-1)]
    for r in range(9): print('   '+'   '.join(x[r] for x in rows))
```