```python
import json, numpy as np
exec(open('/work/go.py').read())
from collections import Counter
c=Counter()
for P in (3,):
    G=json.load(open(f'/work/pass_{P}/games.json'))
    B=np.load(f'/work/pass_{P}/boards.npy')
    for gi,g in enumerate(G):
        me=1 if g['net_color']=='black' else 2; op=3-me
        for m in g['moves']:
            if m['by']!='net' or m['move']==81 or m['ply']==0: continue
            b=np.where(B[gi][m['ply']-1]==-1,2,B[gi][m['ply']-1]).astype(int)
            p=m['move']
            mc=[y for y in NB[p] if b[y]==me]; E=sum(b[y]==0 for y in NB[p])
            oc=[y for y in NB[p] if b[y]==op]
            if E==0 and len(mc)==1 and not any(chain(b,y)[1]=={p} for y in oc):
                b2=b.copy(); b2[p]=me
                ch,l=chain(b2,p)
                # was it played late?
                c[('al%d'%min(len(l),3),'late' if m['ply']>=80 else 'early', 'nopp%d'%len(oc))]+=1
for k,v in sorted(c.items()): print(k,v)
```