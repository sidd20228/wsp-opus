The three-move corner cycle (0,0) → opponent (0,2) → my (0,1) appears three times. Let me see why (0,0) gets about 98% probability.

```python
import json, numpy as np
exec(open('/work/build8.py').read().split("CC,F0O")[0])
d='/work/pass_9/'
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
for gi in [23,111,113,10,69]:
    g=G[gi]; mv=g['moves']; n=len(mv); last=B[gi,n-1]
    k=[j for j in range(n-1) if (B[gi,j]==last).all()][-1]
    e=mv[k+1]['move']; b=B[gi,k]; me=1 if g['net_color']=='black' else -1
    print(gi,g['net_color'],'move',divmod(e,9),mv[k+1]['top5'][:3])
    print('\n'.join(bs(b)[:3]) if e<27 else '\n'.join(bs(b)[5:]))
    for q in NB[e]:
        if b[q]!=0:
            mem,l=chain_of(b,q); print('  nbr',divmod(q,9),'mine' if b[q]==me else 'opp',len(mem),'libs',sorted(divmod(x,9) for x in l))
```