```python
import json, numpy as np
exec(open('/work/an.py').read())
G=json.load(open('/work/pass_2/games.json'))
B=np.load('/work/pass_2/boards.npy'); Vd=np.load('/work/pass_2/valid.npy')
print(G[0].keys())
print(G[0]['moves'][:3])
for gi,g in enumerate(G):
    if g['opponent']=='random' and g['result']<0:
        print(gi,g['net_color'],g['margin'],g['end'],len(g['moves']))
        last=int(Vd[gi].sum())-1
        bb=B[gi][last]; 
        for r in range(9): print(''.join({0:'.',1:'X',-1:'O'}[int(v)] for v in bb[r*9:r*9+9]))
        # net passes
        mv=[(m['ply'],m['move']) for m in g['moves'] if m['by']=='net' and m['move']==81]
        print('net passes at plies',[p for p,_ in mv])
```