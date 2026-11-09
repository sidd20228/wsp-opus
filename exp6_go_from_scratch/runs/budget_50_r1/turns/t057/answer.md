The catastrophic loss is a capturing race: my big center group died while my net played small corner moves — it has no notion of liberty counts beyond atari. To add "≤2 liberty" detection for opponent groups (atari moves) I need L6 capacity. One option is to compute per-stone tactical indicators at L5 and sum them per point in a single L6 unit, at the cost of 1 step of flood radius. Let me measure how much radius 3 vs 4 matters on real game positions.

```python
exec(open('/work/ana.py').read())
from collections import Counter
G=json.load(open('/work/pass_11/games.json')); Bd=np.load('/work/pass_11/boards.npy')
def tob(arr): return [1 if x==1 else (2 if x==-1 else 0) for x in arr]
def star_pl(b,v):
    c=b[v]; return sum(e(b,q) for q in [v]+NB[v] if b[q]==c)
cnt=Counter()
for gi,g in enumerate(G):
    for k in range(5,len(g['moves']),4):
        b=tob(Bd[gi,k])
        for t in range(81):
            if not b[t] or not any(b[q]==0 for q in NB[t]): continue
            gg,l=group(b,t)
            if len(l)<2: continue
            dist=chain_dist(b,t)
            wd=[dist[v] for v in gg if hv_witness(b,v)]
            md=min(wd) if wd else 99
            # flood radius r means witness center within r-? ; AtO at L6 uses witness centers within 4 (L3:1, L4:2, L5:3, L6:4)
            cnt['n']+=1
            for r in (2,3,4): cnt['false_r%d'%r]+= (md>r)
print({k:v for k,v in cnt.items()}, {k:round(v/cnt['n'],4) for k,v in cnt.items()})
```