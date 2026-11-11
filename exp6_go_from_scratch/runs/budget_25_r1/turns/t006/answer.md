Against the strong bot, 30 of 64 games ended by repetition, even though the mean margin was +33. I need to find out who made the repeating move.

```python
import json, numpy as np
d='/work/pass_2/'
G=json.load(open(d+'games.json'))
print(G[0].keys())
B=np.load(d+'boards.npy'); V=np.load(d+'valid.npy')
from collections import Counter
cnt=Counter()
for gi,g in enumerate(G):
    if g['end']!='two_passes' and 'rep' in g['end']:
        last=g['moves'][-1]
        net_last = (last['by']==g['net_color']) if 'by' in last else None
        cnt[(g['opponent'],g['result'],last['by'],g['net_color'])]+=1
print(cnt)
print(G[0]['end'], G[0]['moves'][:3])
```