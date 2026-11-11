```python
import json, numpy as np
d='/work/pass_2/'
G=json.load(open(d+'games.json')); B=np.load(d+'boards.npy')
def bs(b):
    m={0:'.',1:'X',-1:'O'}
    return [''.join(m[int(b[r*9+c])] for c in range(9)) for r in range(9)]
reps=[i for i,g in enumerate(G) if 'rep' in g['end']]
for gi in reps[:3]:
    g=G[gi]; mv=g['moves']; n=len(mv)
    print('game',gi,g['opponent'],g['net_color'],'len',n,'margin',g['margin'])
    for m in mv[-8:]:
        print(m['ply'],m['by'],m['move'],(m.get('top5') or '')[:3] if m.get('top5') else '')
    for k in [n-5,n-3,n-2,n-1]:
        print('after ply',k)
        print('\n'.join(bs(B[gi,k])))
```