Move-feature distributions now look similar to the opponent's, so there's little left there. I'll look instead at big capture events against my net (4+ stones) to find tactical weaknesses.

```python
import json, numpy as np
from collections import Counter
exec(open('/work/build8.py').read().split("CC,F0O")[0])
d='/work/pass_12/'
G=json.load(open(d+'games.json')); B=np.load(d+'boards.npy')
def bs(b):
    m={0:'.',1:'X',-1:'O'}
    return [''.join(m[int(b[r*9+c])] for c in range(9)) for r in range(9)]
events=[]
for gi,g in enumerate(G):
    if g['opponent']!='strong': continue
    me=1 if g['net_color']=='black' else -1; mv=g['moves']
    for k in range(1,len(mv)):
        if mv[k]['by']!='opponent': continue
        lost=int(((B[gi,k-1]==me)&(B[gi,k]==0)).sum())
        if lost>=4: events.append((gi,k,lost,g['result']))
print(len(events), Counter(e[2] for e in events))
for gi,k,lost,res in events[:6]:
    g=G[gi]; mv=g['moves']
    print('game',gi,g['net_color'],'res',res,'ply',k,'lost',lost,'opp move',divmod(mv[k]['move'],9))
    for j in [k-3,k-1]:
        m=mv[j]; print('   net ply',j,divmod(m['move'],9) if m['move']<81 else 'pass',m['top5'][:3])
    print('\n'.join(bs(B[gi,k-2])))
```