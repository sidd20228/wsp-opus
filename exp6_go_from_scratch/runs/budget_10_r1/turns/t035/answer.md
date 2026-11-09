```python
import json, numpy as np
exec(open('/work/go.py').read())
from collections import deque
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
def pb(b):
    for r in range(9): print(''.join('.XO'[v] for v in b[r*9:r*9+9]))
for gi,ply,t in ((0,66,12),(1,78,12),(3,88,43),(4,68,68)):
    g=G[gi]; me=1 if g['net_color']=='black' else 2; op=3-me
    b=np.where(B[gi][ply-1]==-1,2,B[gi][ply-1]).astype(int)
    print(gi,ply,'me','XO'[me-1],'move',divmod(t,9))
    pb(b)
    for y in NB[t]:
        if b[y]==0: continue
        ch,l=chain(b,y); print('  nbr',divmod(y,9),'col',b[y],'size',len(ch),'libs',len(l),'reach3libs',sorted(reach_libs(b,y,3)))
```