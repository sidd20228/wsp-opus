Same totals; game 103 is still lost. Let me see whether the eye-filling move still happens there.

```python
import json, numpy as np
G=json.load(open('/work/pass_31/games.json')); B=np.load('/work/pass_31/boards.npy')
def show(b):
    ch={1:'X',-1:'O',0:'.'}
    return [''.join(ch[int(b[r*9+c])] for c in range(9)) for r in range(9)]
gi=103; g=G[gi]
for j in range(47,66):
    m=g['moves'][j]
    print(j, m['by'], divmod(m['move'],9) if m['move']<81 else 'P', [(divmod(a,9) if a<81 else 'P',round(p,2)) for a,p in m.get('top5',[])[:4]])
rows=[show(B[gi,j]) for j in range(50,66,2)]
for r in range(9): print('  '+' '.join(x[r] for x in rows))
```