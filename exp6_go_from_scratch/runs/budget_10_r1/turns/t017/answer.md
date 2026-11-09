```python
import json, numpy as np
exec(open('/work/go.py').read())
B=np.load('/work/pass_2/boards.npy'); Vd=np.load('/work/pass_2/valid.npy')
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
res={R:[0,0] for R in (2,3,4)}; true=0
for gi in range(B.shape[0]):
    for ply in range(0,int(Vd[gi].sum()),3):
        b=np.where(B[gi][ply]==-1,2,B[gi][ply]).astype(int)
        for p in range(81):
            if b[p]!=0: continue
            for s in NB[p]:
                if b[s]==0: continue
                at = chain(b,s)[1]=={p}
                true+=at
                for R in res:
                    rl = reach_libs(b,s,R)=={p}
                    if rl and not at: res[R][0]+=1
                    if rl: res[R][1]+=1
print('true atari edges',true, {R:(v[0],v[1]) for R,v in res.items()})
```