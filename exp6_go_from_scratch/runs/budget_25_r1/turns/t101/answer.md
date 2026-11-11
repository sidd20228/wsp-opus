Pairing is confirmed: 73 of 128 games were identical, and in the 55 that diverged v8E won 38 against v8F's 32. So small targeted changes can be compared fairly. For one more paired test, I'll extend the cycle detector to the edge points where cycle starts occurred most often, using the 16 remaining spare units. Let me count edge cycle-start locations across all recorded passes.

```python
import json, numpy as np
from collections import Counter
exec(open('/work/build8.py').read().split("CC,F0O")[0])
cnt=Counter(); dirs=Counter()
for pk in range(2,24):
    try:
        G=json.load(open(f'/work/pass_{pk}/games.json')); B=np.load(f'/work/pass_{pk}/boards.npy')
    except Exception: continue
    for gi,g in enumerate(G):
        if 'rep' not in g['end']: continue
        mv=g['moves']; n=len(mv); last=B[gi,n-1]
        ks=[j for j in range(n-1) if (B[gi,j]==last).all()]
        if not ks: continue
        k=ks[-1]
        if mv[k+1]['by']!='net' or n-k-1!=3: continue
        e=mv[k+1]['move']; b=B[gi,k]; me=1 if g['net_color']=='black' else -1
        if line(e)==0 and e not in (0,8,72,80):
            cnt[e]+=1
            own=[q for q in NB[e] if b[q]==me]
            if len(own)==1:
                n_=own[0]; dirs['along' if line(n_)==0 else 'inward']+=1
print(sorted([(divmod(k,9),v) for k,v in cnt.items()],key=lambda x:-x[1]))
print(dirs, 'distinct edge pts',len(cnt))
```