```python
import json, numpy as np
exec(open('/work/go.py').read())
from collections import Counter
G=json.load(open('/work/pass_9/games.json'))
B=np.load('/work/pass_9/boards.npy')
c=Counter()
for gi,g in enumerate(G):
    if g['opponent']!='strong': continue
    me=1 if g['net_color']=='black' else 2; op=3-me
    for m in g['moves']:
        if m['by']!='net' or m['move']!=81: continue
        ply=m['ply']
        b=np.where(B[gi][ply-1]==-1,2,B[gi][ply-1]).astype(int)
        bprev=np.where(B[gi][ply-2]==-1,2,B[gi][ply-2]).astype(int) if ply>=2 else None
        lastm=g['moves'][ply-1]['move']
        for p in range(81):
            if b[p]!=0: continue
            if not any(b[y]==op and chain(b,y)[1]=={p} for y in NB[p]): continue
            # ko-illegal? previous board had our stone at p and opp's last move captured exactly it
            koill = bprev is not None and bprev[p]==me and lastm<81 and lastm in NB[p] and int(((bprev==me)&(b==0)).sum())==1
            no=sum(b[y]==me for y in NB[p]); E=sum(b[y]==0 for y in NB[p])
            b2=b.copy(); b2[p]=me; ncap=0
            for y in NB[p]:
                if b2[y]==op and not chain(b2,y)[1]:
                    ch=chain(b2,y)[0]; ncap+=len(ch)
                    for z in ch: b2[z]=0
            c[('koillegal' if koill else 'legal','own%d'%no,'E%d'%E,'ncap%d'%min(ncap,3))]+=1
for k,v in c.most_common(12): print(v,k)
```