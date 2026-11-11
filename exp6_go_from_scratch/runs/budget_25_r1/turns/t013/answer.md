With four hops, only about 0.5% of liberty/stone pairs lack a witness within range, so false captures should be rare in general. Let me diagnose each cycle-starting move directly.

```python
import json, numpy as np
from collections import Counter, deque
d='/work/pass_2/'
G=json.load(open(d+'games.json')); B=np.load(d+'boards.npy')
def nbrs(p):
    r,c=divmod(p,9); o=[]
    for dr,dc in((-1,0),(1,0),(0,-1),(0,1)):
        rr,cc=r+dr,c+dc
        if 0<=rr<9 and 0<=cc<9:o.append(rr*9+cc)
    return o
NB=[nbrs(p) for p in range(81)]
PAR=[(p//9+p%9)%2 for p in range(81)]
def chain_of(b,p):
    col=b[p]; mem=[p]; s={p}; st=[p]
    while st:
        x=st.pop()
        for q in NB[x]:
            if b[q]==col and q not in s: s.add(q); st.append(q); mem.append(q)
    libs=set(q for x in mem for q in NB[x] if b[q]==0)
    return mem,libs
def witness(b,n,e):
    mem,libs=chain_of(b,n); ms=set(mem)
    dist={n:0}; dq=deque([n])
    while dq:
        x=dq.popleft()
        nl=[q for q in NB[x] if b[q]==0]
        if len(nl)>=2 or any(PAR[q]!=PAR[e] for q in nl): return dist[x]
        for q in NB[x]:
            if q in ms and q not in dist: dist[q]=dist[x]+1; dq.append(q)
    return None
reps=[i for i,g in enumerate(G) if 'rep' in g['end']]
for gi in reps:
    g=G[gi]; mv=g['moves']; n=len(mv)
    last=B[gi,n-1]
    k=[j for j in range(n-1) if (B[gi,j]==last).all()][-1]
    m=mv[k+1]
    if m['by']!='net': print(gi,'starts with opp'); continue
    e=m['move']; b=B[gi,k].copy()
    me = 1 if g['net_color']=='black' else -1
    if e==81: print(gi,'pass'); continue
    info=[]
    for q in NB[e]:
        if b[q]==-me:
            mem,libs=chain_of(b,q); info.append(('opp',len(mem),len(libs),witness(b,q,e)))
        elif b[q]==me:
            mem,libs=chain_of(b,q); info.append(('own',len(mem),len(libs)))
        else: info.append('empty')
    print(gi,'move',divmod(e,9),'p=',m['top5'][0], info)
```