```python
import json, numpy as np
exec(open('/work/go.py').read())
from collections import deque, Counter
def reach_libs(b,s,R):
    col=b[s]; dist={s:0}; dq=deque([s]); libs=set()
    while dq:
        x=dq.popleft()
        for y in NB[x]:
            if b[y]==0: libs.add(y)
            elif b[y]==col and y not in dist and dist[x]<R:
                dist[y]=dist[x]+1; dq.append(y)
    return libs
G=json.load(open('/work/pass_4/games.json'))
B=np.load('/work/pass_4/boards.npy')
c=Counter(); nm=0
for gi,g in enumerate(G):
    me=1 if g['net_color']=='black' else 2; op=3-me
    for m in g['moves']:
        if m['by']!='net' or m['move']==81: continue
        nm+=1; ply=m['ply']
        if ply==0: continue
        b=np.where(B[gi][ply-1]==-1,2,B[gi][ply-1]).astype(int)
        t=m['move']
        oc=[y for y in NB[t] if b[y]==op]
        real=any(chain(b,y)[1]=={t} for y in oc)
        for R in (3,4):
            fake=(not real) and any(reach_libs(b,y,R)=={t} for y in oc)
            if fake: c[R]+=1
        if real: c['real']+=1
print(nm,c)
```