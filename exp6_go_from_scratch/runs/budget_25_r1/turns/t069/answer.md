v13 is about the same (67.5% / 69%), but repetition losses jumped to 16: 8 against each opponent. Let me see which cycle types dominate now.

```python
import json, numpy as np
from collections import Counter
exec(open('/work/build8.py').read().split("CC,F0O")[0])
def chain_of(b,p):
    col=b[p]; mem=[p]; s={p}; st=[p]
    while st:
        x=st.pop()
        for q in NB[x]:
            if b[q]==col and q not in s: s.add(q); st.append(q); mem.append(q)
    libs=set(q for x in mem for q in NB[x] if b[q]==0)
    return mem,libs
for d in ['/work/pass_12/','/work/pass_13/']:
    G=json.load(open(d+'games.json')); B=np.load(d+'boards.npy')
    print(d)
    for gi,g in enumerate(G):
        if 'rep' not in g['end']: continue
        mv=g['moves']; n=len(mv); last=B[gi,n-1]
        k=[j for j in range(n-1) if (B[gi,j]==last).all()][-1]
        seq=[(m['by'][0],m['move'],(m['top5'][0][1] if m.get('top5') else None)) for m in mv[k+1:]]
        info=None; m=mv[k+1]
        if m['by']=='net' and m['move']<81:
            e=m['move']; b=B[gi,k]; me=1 if g['net_color']=='black' else -1
            info=[]
            for q in NB[e]:
                if b[q]==-me: mem,l=chain_of(b,q); info.append(('x',len(mem),len(l)))
                elif b[q]==me: mem,l=chain_of(b,q); info.append(('o',len(mem),len(l)))
                else: info.append('.')
        print(' ',gi,g['opponent'][:2],len(seq),seq[:5],info)
```