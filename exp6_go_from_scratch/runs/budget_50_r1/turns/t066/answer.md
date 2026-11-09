Same 94.5%, but now White is 100% and all 7 losses are as Black. Let me look at those losses.

```python
import json, numpy as np
from collections import Counter
G=json.load(open('/work/pass_14/games.json')); B=np.load('/work/pass_14/boards.npy')
print(Counter((g['end'],g['result']) for g in G))
def show(b):
    ch={1:'X',-1:'O',0:'.'}
    return [''.join(ch[int(b[r*9+c])] for c in range(9)) for r in range(9)]
for gi,g in enumerate(G):
    if g['result']==-1:
        n=len(g['moves'])
        s=f"{gi} {g['net_color']} {g['end']} margin {g['margin']} len {n} cap {g['stones_captured_by_net']} lost {g['stones_lost_by_net']}"
        if g['end']=='repetition':
            last=B[gi,n-1]; k0=[k for k in range(n-1) if np.array_equal(B[gi,k],last)][-1]
            s+=' cycle '+str([(m['by'][0], divmod(m['move'],9) if m['move']<81 else 'P', round(m['top5'][0][1],2) if 'top5' in m else '') for m in g['moves'][k0+1:n]][:8])
        print(s)
# opening moves distribution for net as black
from collections import Counter
op=Counter(tuple(divmod(g['moves'][0]['move'],9)) for g in G if g['net_color']=='black')
print(op)
op2=Counter((tuple(divmod(g['moves'][1]['move'],9)), tuple(divmod(g['moves'][2]['move'],9))) for g in G if g['net_color']=='black')
print(op2.most_common(10))
```