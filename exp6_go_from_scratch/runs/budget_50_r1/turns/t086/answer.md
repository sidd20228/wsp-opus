No improvement (Black 87.5%) — the earlier split was likely noise. Let me confirm the openings were applied and look at how Black's losses arise now.

```python
import json
from collections import Counter
G=json.load(open('/work/pass_19/games.json'))
print(Counter(divmod(g['moves'][0]['move'],9) for g in G if g['net_color']=='black'))
print(Counter((divmod(g['moves'][1]['move'],9), g['result']) for g in G if g['net_color']=='black').most_common(12))
print([(gi,g['margin'],g['end']) for gi,g in enumerate(G) if g['result']==-1])
```