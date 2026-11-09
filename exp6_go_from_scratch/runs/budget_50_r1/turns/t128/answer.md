```python
import json
A=json.load(open('/work/pass_30/games.json')); Bv=json.load(open('/work/pass_31/games.json'))
from collections import Counter
print(Counter((a['result'],b['result']) for a,b in zip(A,Bv)))
for i,(a,b) in enumerate(zip(A,Bv)):
    if a['result']!=b['result'] or b['result']==-1: print(i,a['net_color'],'v26',a['result'],a['margin'],a['end'],'| v27',b['result'],b['margin'],b['end'], b['stones_lost_by_net'])
```