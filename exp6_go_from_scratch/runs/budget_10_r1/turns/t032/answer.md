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
def after_libs(b,p,me):
    b2=b.copy(); b2[p]=me
    op=3-me
    for y in NB[p]:
        if b2[y]==op and not chain(b2,y)[1]:
            for z in chain(b2,y)[0]: b2[z]=0
    return len(chain(b2,p)[1])
B=np.load('/work/pass_4/boards.npy'); Vd=np.load('/work/pass_4/valid.npy')
cf=Counter(); ct=Counter()
for gi in range(0,B.shape[0],2):
    n=int(Vd[gi].sum())
    for ply in range(0,n,3):
        b=np.where(B[gi][ply]==-1,2,B[gi][ply]).astype(int)
        for col in (1,2):
            me=3-col
            for p in range(81):
                if b[p]!=0: continue
                oc=[y for y in NB[p] if b[y]==col]
                if not oc: continue
                real=any(chain(b,y)[1]=={p} for y in oc)
                fake=any(reach_libs(b,y,3)=={p} for y in oc)
                if not fake: continue
                E=sum(b[y]==0 for y in NB[p])
                al=after_libs(b,p,me)
                (ct if real else cf)[(min(E,2),min(al,2))]+=1
print('true',sorted(ct.items())); print('false',sorted(cf.items()))
```