Variant B is clearly better: 74% against strong (from 60%) and 79% against weak, with a mean margin of +48. Let me look at the remaining losses under B.

```python
import json, numpy as np
from collections import Counter
exec(open('/work/build8.py').read().split("CC,F0O")[0])
d='/work/pass_9/'
G=json.load(open(d+'games.json')); B=np.load(d+'boards.npy')
def bs(b):
    m={0:'.',1:'X',-1:'O'}
    return [''.join(m[int(b[r*9+c])] for c in range(9)) for r in range(9)]
for gi,g in enumerate(G):
    if g['result']>0: continue
    n=len(g['moves'])
    s=f"{gi} {g['opponent'][:2]} {g['net_color'][0]} {g['end']} mg {g['margin']} len {n} cap {g['stones_captured_by_net']} lost {g['stones_lost_by_net']}"
    if 'rep' in g['end']:
        mv=g['moves']; last=B[gi,n-1]
        k=[j for j in range(n-1) if (B[gi,j]==last).all()][-1]
        seq=[(m['by'][0],m['move'],(m['top5'][0][1] if m.get('top5') else None)) for m in mv[k+1:]]
        s+=' seq '+str(seq)
    print(s)
gi=[i for i,g in enumerate(G) if g['result']<0 and 'two' in g['end'] and g['opponent']=='strong'][0]
g=G[gi]; print(gi,g['net_color'])
for k in [9,19,29,39,len(g['moves'])-1]:
    print('ply',k); print('\n'.join(bs(B[gi,k])))
```