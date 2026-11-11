The dominant repetition is still the three-move cycle that starts with an E=0 connection to my own single two-liberty stone. I proved that no single linear self-atari unit can block it without also blocking cut-point connections or eye-filling kills, because the fake-capture count interferes. The cleanest fix is a dedicated unit Z: E=0, exactly one own neighbour, and no own stones at Manhattan distance 2 (a lone stone being connected). Let me check how well "no own stones at distance 2" identifies these cycle starts.

```python
import json, numpy as np
from collections import Counter
exec(open('/work/build8.py').read().split("CC,F0O")[0])
def md(p,q): return abs(p//9-q//9)+abs(p%9-q%9)
D2=[[q for q in range(81) if md(p,q)==2] for p in range(81)]
def chain_of(b,p):
    col=b[p]; mem=[p]; s={p}; st=[p]
    while st:
        x=st.pop()
        for q in NB[x]:
            if b[q]==col and q not in s: s.add(q); st.append(q); mem.append(q)
    libs=set(q for x in mem for q in NB[x] if b[q]==0)
    return mem,libs
cyc=Counter(); norm=Counter()
for pk in [9,10,11,12,13]:
    d=f'/work/pass_{pk}/'
    G=json.load(open(d+'games.json')); B=np.load(d+'boards.npy')
    starts=set()
    for gi,g in enumerate(G):
        if 'rep' not in g['end']: continue
        mv=g['moves']; n=len(mv); last=B[gi,n-1]
        k=[j for j in range(n-1) if (B[gi,j]==last).all()][-1]
        if mv[k+1]['by']=='net' and len(mv)-k-1==3: starts.add((gi,k+1))
    for gi,g in enumerate(G):
        me=1 if g['net_color']=='black' else -1
        for m in g['moves']:
            if m['by']!='net' or m['move']==81: continue
            k=m['ply']; e=m['move']; b=B[gi,k-1]
            so=[q for q in NB[e] if b[q]==me]; E=sum(b[q]==0 for q in NB[e])
            if E!=0 or len(so)!=1: continue
            d2=sum(b[q]==me for q in D2[e])
            mem,l=chain_of(b,so[0])
            key=(min(d2,3),'single' if len(mem)==1 else 'multi', len(l))
            if (gi,k) in starts: cyc[key]+=1
            else: norm[key]+=1
print('cycle starts',sorted(cyc.items()))
print('other E=0 s_o=1 moves',sorted(norm.items()))
```