The main cycle starter is a move with no empty neighbours that connects to my single stone with two liberties. My self-atari detector misses it because it can't tell two liberties from three. A few cycles are double-ko loops triggered by my late-game passes. Before redesigning, let me look at the non-repetition losses.

```python
import json, numpy as np
d='/work/pass_2/'
G=json.load(open(d+'games.json')); B=np.load(d+'boards.npy')
def bs(b):
    m={0:'.',1:'X',-1:'O'}
    return [''.join(m[int(b[r*9+c])] for c in range(9)) for r in range(9)]
for gi,g in enumerate(G):
    if g['result']<0 and 'rep' not in g['end']:
        n=len(g['moves'])
        print(gi,g['opponent'],g['net_color'],g['end'],'margin',g['margin'],'len',n,'cap',g['stones_captured_by_net'],'lost',g['stones_lost_by_net'])
        print('\n'.join(bs(B[gi,n-1])))
ms=[g['margin'] for g in G if g['opponent']=='strong' and 'rep' not in g['end']]
print(sorted(ms))
```