```python
import json, numpy as np
G=json.load(open('/work/pass_5/games.json'))
B=np.load('/work/pass_5/boards.npy'); Vd=np.load('/work/pass_5/valid.npy')
from collections import Counter
print(Counter((g['opponent'],g['end'],g['result']) for g in G))
def pb(bb):
    for r in range(9): print(''.join({0:'.',1:'X',-1:'O'}[int(v)] for v in bb[r*9:r*9+9]))
k=0
for gi,g in enumerate(G):
    if g['end']=='two passes' and g['result']<0 and k<3:
        k+=1; n=int(Vd[gi].sum())
        netp=[m['ply'] for m in g['moves'] if m['by']=='net' and m['move']==81]
        print(gi,g['net_color'],n,'margin',g['margin'],'first net pass',netp[:3],len(netp))
        fp=netp[0]
        print('board at first pass'); pb(B[gi][fp]); print('final'); pb(B[gi][n-1])
        m=g['moves'][fp]; print(m['top5'])
```