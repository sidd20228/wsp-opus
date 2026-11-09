Worse (94.5%). Since game seeds seem fixed per index, let me do a paired comparison of per-game outcomes between v24 (pass 27) and this variant to gauge how correlated the passes are.

```python
import json
A=json.load(open('/work/pass_27/games.json')); Bv=json.load(open('/work/pass_29/games.json'))
same_open=sum(1 for a,b in zip(A,Bv) if a['moves'][0]['move']==b['moves'][0]['move'] and a['moves'][1]['move']==b['moves'][1]['move'])
both=[(a['result'],b['result']) for a,b in zip(A,Bv)]
from collections import Counter
print('same first 2 moves:',same_open, Counter(both))
# check opponent's first move for same game index across passes 25,26,27
C=json.load(open('/work/pass_25/games.json'))
print(sum(1 for a,c in zip(A,C) if a['net_color']==c['net_color'] and a['moves'][0]['move']==c['moves'][0]['move']))
```