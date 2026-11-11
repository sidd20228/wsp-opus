v8F drops to 91/128 from v8E's 97, so the snapback change doesn't help; v8E stays the candidate. One consideration: if the practice games use fixed random streams, two networks differing only in rare situations play identical games until the first divergence. That would make small A/B tests paired and more informative. Let me check that between v8E and v8F.

```python
import json, numpy as np
GE=json.load(open('/work/pass_22/games.json')); GF=json.load(open('/work/pass_23/games.json'))
same=0; diverge=[]
for a,b in zip(GE,GF):
    ma=[m['move'] for m in a['moves']]; mb=[m['move'] for m in b['moves']]
    if ma==mb: same+=1
    else:
        k=next(i for i,(x,y) in enumerate(zip(ma,mb)) if x!=y) if any(x!=y for x,y in zip(ma,mb)) else min(len(ma),len(mb))
        diverge.append((k,a['result'],b['result']))
print('identical games',same,'of',len(GE))
print('divergent games: E wins',sum(1 for k,r1,r2 in diverge if r1>0),'F wins',sum(1 for k,r1,r2 in diverge if r2>0),'n',len(diverge))
print('first divergence plies',sorted(k for k,_,_ in diverge)[:40])
```