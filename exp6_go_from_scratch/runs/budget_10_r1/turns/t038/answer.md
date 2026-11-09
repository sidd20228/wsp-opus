```python
import json, numpy as np
exec(open('/work/go.py').read())
G=json.load(open('/work/pass_4/games.json'))
B=np.load('/work/pass_4/boards.npy'); Vd=np.load('/work/pass_4/valid.npy')
from collections import Counter
c=Counter(); ex=[]
for gi,g in enumerate(G):
    if g['opponent']!='strong': continue
    me=1 if g['net_color']=='black' else 2; op=3-me
    n=int(Vd[gi].sum())
    for m in g['moves']:
        if m['by']!='net' or m['ply']==0: continue
        ply=m['ply']
        b=np.where(B[gi][ply-1]==-1,2,B[gi][ply-1]).astype(int)
        seen=set()
        for s in range(81):
            if b[s]!=me or s in seen: continue
            ch,l=chain(b,s); seen|=ch
            if len(l)==1 and len(ch)>=3:
                lib=list(l)[0]
                E=sum(b[y]==0 for y in NB[lib])
                # after net move and opp move, chain alive?
                if ply+1<n:
                    b2=np.where(B[gi][ply+1]==-1,2,B[gi][ply+1]).astype(int)
                    alive=all(b2[x]==me for x in ch)
                else: alive=True
                kind='extended' if m['move']==lib else 'other'
                c[(kind,alive,min(E,3))]+=1
                if kind=='other' and len(ex)<4: ex.append((gi,ply,divmod(lib,9),len(ch),m['top5'][:3]))
print(c)
for e in ex: print(e)
```