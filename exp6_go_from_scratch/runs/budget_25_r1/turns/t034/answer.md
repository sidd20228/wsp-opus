Still worse: 25 repetitions, and passes per game unchanged even with the pass logit at −16. That number may count both players' passes. I'll analyse the cycles and who passes.

```python
import json, numpy as np
from collections import Counter, deque
exec(open('/work/build6.py').read().split("# L1-L4 layout")[0])
d='/work/pass_6/'
G=json.load(open(d+'games.json')); B=np.load(d+'boards.npy')
def chain_of(b,p):
    col=b[p]; mem=[p]; s={p}; st=[p]
    while st:
        x=st.pop()
        for q in NB[x]:
            if b[q]==col and q not in s: s.add(q); st.append(q); mem.append(q)
    libs=set(q for x in mem for q in NB[x] if b[q]==0)
    return mem,libs
npass=Counter()
for g in G:
    for m in g['moves']:
        if m['move']==81: npass[(g['opponent'],m['by'])]+=1
print(npass)
reps=[i for i,g in enumerate(G) if 'rep' in g['end']]
for gi in reps:
    g=G[gi]; mv=g['moves']; n=len(mv)
    last=B[gi,n-1]
    k=[j for j in range(n-1) if (B[gi,j]==last).all()][-1]
    seq=[(m['by'][0],m['move'],(m['top5'][0][1] if m.get('top5') else None)) for m in mv[k+1:]]
    m=mv[k+1]; info=None
    if m['by']=='net' and m['move']<81:
        e=m['move']; b=B[gi,k]; me=1 if g['net_color']=='black' else -1
        info=[]
        for q in NB[e]:
            if b[q]==-me: mem,l=chain_of(b,q); info.append(('x',len(mem),len(l)))
            elif b[q]==me: mem,l=chain_of(b,q); info.append(('o',len(mem),len(l)))
            else: info.append('.')
    print(gi,g['opponent'][:2],g['net_color'][0],'len',n,'mg',g['margin'],'seq',seq,info)
```