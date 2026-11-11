```python
import json, numpy as np
exec(open('/work/go.py').read())
from collections import Counter
G=json.load(open('/work/pass_3/games.json'))
B=np.load('/work/pass_3/boards.npy')
c=Counter()
for gi,g in enumerate(G):
    me=1 if g['net_color']=='black' else 2; op=3-me
    for m in g['moves']:
        if m['by']!='net' or m['move']==81 or m['ply']==0: continue
        b=np.where(B[gi][m['ply']-1]==-1,2,B[gi][m['ply']-1]).astype(int)
        p=m['move']
        mc=[y for y in NB[p] if b[y]==me]
        E=sum(b[y]==0 for y in NB[p])
        if mc and all(chain(b,y)[1]=={p} for y in mc):
            cap=any(b[y]==op and chain(b,y)[1]=={p} for y in NB[p])
            c[('OAcase','E%d'%min(E,2),'cap' if cap else 'nocap')]+=1
        oc=[y for y in NB[p] if b[y]==op]
        E0own1 = E==0 and len(mc)==1 and not any(b[y]==op and chain(b,y)[1]=={p} for y in oc)
        if E0own1: c['E0own1_nc']+=1
        c['all']+=1
print(c)
```