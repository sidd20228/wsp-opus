v8A is the best so far: 60% against strong (up from 30–40%), with 49 captures per game. Against weak it's 73%. Before testing the prior variant, I'll look at the weak losses and the remaining repetitions.

```python
import json, numpy as np
from collections import Counter
exec(open('/work/build8.py').read().split("CC,F0O")[0])
d='/work/pass_8/'
G=json.load(open(d+'games.json')); B=np.load(d+'boards.npy')
def bs(b):
    m={0:'.',1:'X',-1:'O'}
    return [''.join(m[int(b[r*9+c])] for c in range(9)) for r in range(9)]
def chain_of(b,p):
    col=b[p]; mem=[p]; s={p}; st=[p]
    while st:
        x=st.pop()
        for q in NB[x]:
            if b[q]==col and q not in s: s.add(q); st.append(q); mem.append(q)
    libs=set(q for x in mem for q in NB[x] if b[q]==0)
    return mem,libs
for gi,g in enumerate(G):
    if g['result']>0: continue
    n=len(g['moves'])
    s=f"{gi} {g['opponent'][:2]} {g['net_color'][0]} {g['end']} mg {g['margin']} len {n} cap {g['stones_captured_by_net']} lost {g['stones_lost_by_net']}"
    if 'rep' in g['end']:
        mv=g['moves']; last=B[gi,n-1]
        k=[j for j in range(n-1) if (B[gi,j]==last).all()][-1]
        seq=[(m['by'][0],m['move'],(m['top5'][0][1] if m.get('top5') else None)) for m in mv[k+1:]]
        s+=' seq '+str(seq)
    print(s)
```