Good start (64% vs strong, 88% vs weak). But 14 games vs strong ended by repetition — let me see who caused them and how losses happen.

```python
import json, numpy as np
G=json.load(open('/work/pass_2/games.json'))
print(type(G), len(G)); print(G[0].keys())
from collections import Counter
c=Counter((g['opponent'],g['end'],g['result']) for g in G)
for k,v in sorted(c.items()): print(k,v)
for g in G:
    if g['end']!='two_passes' and g['opponent']=='strong':
        mv=g['moves']; print(g['net_color'], g['result'], g['margin'], g['end'], len(mv), [ (m['by'],m['move']) for m in mv[-8:]])
```