```python
import json, numpy as np
G=json.load(open('/work/pass_8/games.json'))
B=np.load('/work/pass_8/boards.npy'); Vd=np.load('/work/pass_8/valid.npy')
from collections import Counter
print(Counter((g['opponent'],g['end'],g['result']) for g in G))
# for losses: stones lost after our first pass
for gi,g in enumerate(G):
    if g['opponent']!='strong' or g['end']!='two passes' or g['result']>0: continue
    n=int(Vd[gi].sum()); netc=1 if g['net_color']=='black' else -1
    netp=[m['ply'] for m in g['moves'] if m['by']=='net' and m['move']==81]
    if not netp: continue
    fp=netp[0]
    own_at_fp=int((B[gi][fp]==netc).sum()); own_end=int((B[gi][n-1]==netc).sum())
    print(gi,'first pass',fp,'n',n,'#passes',len(netp),'own stones at fp',own_at_fp,'end',own_end,'margin',g['margin'])
```