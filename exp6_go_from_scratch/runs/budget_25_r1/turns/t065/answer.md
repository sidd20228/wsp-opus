v12 scored 67.5% against strong and 71% against weak. That's within noise of v8B but not better; pooled, v8B and v12 sit around 70/75. I need larger effects. Let me redo the move-feature comparison against the strong opponent on the v12 games, which use the new prior.

```python
import json, numpy as np
from collections import Counter
exec(open('/work/build8.py').read().split("CC,F0O")[0])
d='/work/pass_12/'
G=json.load(open(d+'games.json')); B=np.load(d+'boards.npy')
DIAG=[[q for q in range(81) if abs(q//9-p//9)==1 and abs(q%9-p%9)==1] for p in range(81)]
stats={'net':Counter(),'opp':Counter()}; n_moves=Counter()
for gi,g in enumerate(G):
    if g['opponent']!='strong': continue
    netc=1 if g['net_color']=='black' else -1
    prev=None
    for m in g['moves']:
        k=m['ply']; e=m['move']
        if e==81: prev=None; continue
        who='net' if m['by']=='net' else 'opp'
        me = netc if who=='net' else -netc
        b=B[gi,k-1] if k>0 else np.zeros(81,int)
        S=int((b!=0).sum())
        phase='early' if S<10 else ('mid' if S<35 else 'late')
        so=sum(b[n]==me for n in NB[e]); sx=sum(b[n]==-me for n in NB[e])
        do=sum(b[n]==me for n in DIAG[e]); dx=sum(b[n]==-me for n in DIAG[e])
        near_last = prev is not None and abs(prev//9-e//9)+abs(prev%9-e%9)<=2
        n_moves[(who,phase)]+=1; st=stats[who]
        st[(phase,'line',line(e))]+=1; st[(phase,'so',so)]+=1; st[(phase,'sx',sx)]+=1
        st[(phase,'do',do)]+=1; st[(phase,'dx',dx)]+=1; st[(phase,'near',int(near_last))]+=1
        st[(phase,'pure_diag',int(so==0 and do>0))]+=1
        prev=e
for phase in ['early','mid','late']:
    print('=== ',phase,'moves net',n_moves[('net',phase)],'opp',n_moves[('opp',phase)])
    for f,rng in [('line',range(5)),('so',range(5)),('sx',range(5)),('do',range(5)),('dx',range(5)),('near',range(2)),('pure_diag',range(2))]:
        row=[]
        for who in ['net','opp']:
            tot=n_moves[(who,phase)]
            row.append(' '.join(f'{stats[who][(phase,f,v)]/max(tot,1):.2f}' for v in rng))
        print(f'{f:9s} net[{row[0]}]  opp[{row[1]}]')
```