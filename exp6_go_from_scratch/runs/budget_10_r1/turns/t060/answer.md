```python
import json, numpy as np
exec(open('/work/go.py').read())
from collections import Counter
G=json.load(open('/work/pass_9/games.json'))
B=np.load('/work/pass_9/boards.npy')
c=Counter(); npass=0
for gi,g in enumerate(G):
    if g['opponent']!='strong': continue
    me=1 if g['net_color']=='black' else 2; op=3-me
    for m in g['moves']:
        if m['by']!='net' or m['move']!=81: continue
        npass+=1
        b=np.where(B[gi][m['ply']-1]==-1,2,B[gi][m['ply']-1]).astype(int)
        g_=Game(); g_.b=b.copy(); g_.turn=me
        kinds=set()
        for p in g_.legal_moves():
            E=sum(b[y]==0 for y in NB[p]); no=sum(b[y]==me for y in NB[p]); nop=sum(b[y]==op for y in NB[p])
            cap=any(b[y]==op and chain(b,y)[1]=={p} for y in NB[p])
            if no==len(NB[p]): k='eye'
            elif cap: k='cap'
            elif E==0: k='E0'
            elif E==1: k='E1own%d'%min(no,1)
            else: k='E2+'
            kinds.add(k)
        c[tuple(sorted(kinds))]+=1
print(npass); 
for k,v in c.most_common(12): print(v,k)
```