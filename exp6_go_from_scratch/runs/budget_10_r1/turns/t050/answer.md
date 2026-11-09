```python
import json, numpy as np
from collections import Counter
G=json.load(open('/work/pass_7/games.json'))
print(Counter((g['opponent'],g['end'],g['result']) for g in G))
m=[g['margin'] for g in G if g['opponent']=='strong' and g['end']=='repetition']
print('rep margins', sorted(m))
```