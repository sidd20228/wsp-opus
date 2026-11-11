The opening is close to random: the board is empty, so a flat prior decides, with move probabilities around 0.1–0.25. My net scatters stones in the center while White takes the edges with second-line moves. Let me look at the strong opponent's opening moves more concretely.

```python
import json, numpy as np
from collections import Counter
d='/work/pass_7/'
G=json.load(open(d+'games.json')); B=np.load(d+'boards.npy')
first=Counter(); seqs=[]
for gi,g in enumerate(G):
    if g['opponent']!='strong': continue
    om=[m['move'] for m in g['moves'] if m['by']=='opponent'][:6]
    if g['net_color']=='white': first[om[0]]+=1
    seqs.append((g['net_color'][0],[divmod(x,9) if x<81 else 'P' for x in om]))
print('opp first move as black:',[(divmod(k,9),v) for k,v in first.most_common()])
for s in seqs[:12]: print(s)
```