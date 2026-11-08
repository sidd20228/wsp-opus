Worse — the net now passes far too often (18 passes/game, 43 games with a pass before move 60), and repetition losses persist. Let me examine both.

```python
import json, numpy as np
from collections import Counter
G=json.load(open('/work/pass_4/games.json')); B=np.load('/work/pass_4/boards.npy')
def show(b):
    ch={1:'X',-1:'O',0:'.'}
    return [''.join(ch[int(b[r*9+c])] for c in range(9)) for r in range(9)]
# first net pass ply distribution
fp=[]
for g in G:
    for m in g['moves']:
        if m['by']=='net' and m['move']==81: fp.append(m['ply']); break
print('first net pass ply', sorted(fp)[:40])
for gi,g in enumerate(G):
    if g['end']!='repetition': continue
    n=len(g['moves']); last=B[gi,n-1]
    k0=[k for k in range(n-1) if np.array_equal(B[gi,k],last)][-1]
    seq=[(m['by'][0], divmod(m['move'],9) if m['move']<81 else 'P') for m in g['moves'][k0+1:n]]
    print(gi,g['opponent'],g['margin'],'cycle len',n-1-k0, seq[:8])
```