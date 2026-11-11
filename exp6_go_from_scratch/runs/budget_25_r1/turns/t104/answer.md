v8G has the same totals as v8E (97/128) with fewer repetitions (3 against 5). Let me look at the paired comparison.

```python
import json, numpy as np
GE=json.load(open('/work/pass_22/games.json')); GG=json.load(open('/work/pass_24/games.json'))
same=0; div=[]
for a,b in zip(GE,GG):
    ma=[m['move'] for m in a['moves']]; mb=[m['move'] for m in b['moves']]
    if ma==mb: same+=1
    else: div.append((a['opponent'],a['result'],b['result'],a['end'],b['end']))
print('identical',same,'divergent',len(div))
for x in div: print(x)
```