```python
import json, numpy as np
G=json.load(open('/work/pass_4/games.json'))
B=np.load('/work/pass_4/boards.npy'); Vd=np.load('/work/pass_4/valid.npy')
from collections import Counter
c=Counter((g['opponent'],g['end'],g['result'],g['moves'][-1]['by']) for g in G); print(c)
def pb(bb):
    for r in range(9): print(''.join({0:'.',1:'X',-1:'O'}[int(v)] for v in bb[r*9:r*9+9]))
k=0
for gi,g in enumerate(G):
    if g['end']=='repetition' and k<2:
        k+=1; n=int(Vd[gi].sum())
        print(gi,g['net_color'],n,'margin',g['margin'])
        for m in g['moves'][-10:]:
            print(m['ply'],m['by'],divmod(m['move'],9) if m['move']<81 else 'pass', [(divmod(a,9) if a<81 else 'P',p) for a,p in m.get('top5',[])[:3]])
        pb(B[gi][n-1])
```