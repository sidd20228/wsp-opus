That confirms it: (2,8) was in atari, and (1,8) would have connected and saved everything, but it looked like an eye. The eye/near-eye unit needs to switch off when an adjacent own chain is in atari.

The boards also show my net building thin diagonal, checkerboard-like shapes. I'll compare simple local statistics of the strong opponent's moves with mine, to inform the positional heuristics (analysis only, no fitting).

```python
import json, numpy as np
from collections import Counter
exec(open('/work/build7.py').read().split("CC,F0O")[0])
d='/work/pass_7/'
G=json.load(open(d+'games.json')); B=np.load(d+'boards.npy')
DIAG=[[q for q in range(81) if abs(q//9-p//9)==1 and abs(q%9-p%9)==1] for p in range(81)]
stats={'net':Counter(),'opp':Counter()}
n_moves=Counter()
for gi,g in enumerate(G):
    if g['opponent']!='strong': continue
    netc=1 if g['net_color']=='black' else -1
    for m in g['moves']:
        k=m['ply']; e=m['move']
        if e==81: continue
        who='net' if m['by']=='net' else 'opp'
        me = netc if who=='net' else -netc
        b=B[gi,k-1] if k>0 else np.zeros(81,int)
        S=int((b!=0).sum())
        if S<10: phase='early'
        elif S<35: phase='mid'
        else: phase='late'
        so=sum(b[n]==me for n in NB[e]); sx=sum(b[n]==-me for n in NB[e])
        do=sum(b[n]==me for n in DIAG[e]); dx=sum(b[n]==-me for n in DIAG[e])
        n_moves[(who,phase)]+=1
        st=stats[who]
        st[(phase,'line',line(e))]+=1
        st[(phase,'so',so)]+=1; st[(phase,'sx',sx)]+=1
        st[(phase,'do',do)]+=1; st[(phase,'dx',dx)]+=1
for phase in ['early','mid','late']:
    print('=== ',phase,'moves net',n_moves[('net',phase)],'opp',n_moves[('opp',phase)])
    for f,rng in [('line',range(5)),('so',range(5)),('sx',range(5)),('do',range(5)),('dx',range(5))]:
        row=[]
        for who in ['net','opp']:
            tot=n_moves[(who,phase)]
            row.append(' '.join(f'{stats[who][(phase,f,v)]/max(tot,1):.2f}' for v in rng))
        print(f'{f:5s} net[{row[0]}]  opp[{row[1]}]')
```