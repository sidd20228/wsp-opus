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
B=np.load('/work/pass_4/boards.npy'); Vd=np.load('/work/pass_4/valid.npy')
st={R:dict(or_true=0,or_false=0,and_true=0,and_false=0) for R in (3,4)}
truecap=0
for gi in range(B.shape[0]):
    n=int(Vd[gi].sum())
    for ply in range(0,n,2):
        b=np.where(B[gi][ply]==-1,2,B[gi][ply]).astype(int)
        for col in (1,2):
            for p in range(81):
                if b[p]!=0: continue
                oc=[y for y in NB[p] if b[y]==col]
                if not oc: continue
                real=any(chain(b,y)[1]=={p} for y in oc)
                truecap+=real
                for R in st:
                    t=[reach_libs(b,y,R)=={p} for y in oc]
                    if any(t): st[R]['or_true' if real else 'or_false']+=1
                    if all(t): st[R]['and_true' if real else 'and_false']+=1
print(truecap, st)
```