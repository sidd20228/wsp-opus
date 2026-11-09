Not better (92.2%); I'll revert to v14. Since only Black loses, let me check whether Black's results depend on its (sampled) first move, pooling the three passes with equivalent nets.

```python
import json
from collections import defaultdict
stats=defaultdict(lambda:[0,0])
for pk in [13,14,17]:
    G=json.load(open('/work/pass_%d/games.json'%pk))
    for g in G:
        if g['net_color']!='black': continue
        m=divmod(g['moves'][0]['move'],9)
        stats[m][0]+=1; stats[m][1]+=(g['result']==1)
for m,(n,w) in sorted(stats.items(), key=lambda x:-x[1][0]): print(m,n,w, round(w/n,2))
exec(open('/work/build6.py').read())
cfg=json.load(open('/work/cfg_v14.json'))
arrs,layout=build6(cfg,inp)
for k,v in arrs.items(): save(k,v)
print('restored v14')
```