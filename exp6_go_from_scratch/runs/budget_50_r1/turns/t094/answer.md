Edge normalization hurt (88.3%); reverting to v14. Let me pool all v14-like passes to categorize the remaining losses.

```python
import json, numpy as np
from collections import Counter
exec(open('/work/build6.py').read())
cfg=json.load(open('/work/cfg_v14.json'))
arrs,layout=build6(cfg,inp)
for k,v in arrs.items(): save(k,v)
c=Counter()
for pk in [13,14,17,19]:
    G=json.load(open('/work/pass_%d/games.json'%pk))
    for g in G:
        c[(g['net_color'],g['end'],g['result'])]+=1
for k in sorted(c): print(k,c[k])
```