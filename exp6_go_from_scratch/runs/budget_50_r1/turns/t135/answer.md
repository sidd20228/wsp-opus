```python
import json
from collections import Counter
A=json.load(open('/work/pass_32/games.json')); Bv=json.load(open('/work/pass_33/games.json'))
print(Counter((a['result'],b['result']) for a,b in zip(A,Bv)))
for i,(a,b) in enumerate(zip(A,Bv)):
    if a['result']!=b['result'] or b['result']==-1: print(i,a['net_color'],'v28',a['result'],a['margin'],a['end'],'| v29',b['result'],b['margin'],b['end'])
```