Repetition losses are now all double-ko 5-cycles that hinge on my pass. Let me look at the position where the net passed in one of them.

```python
import json, numpy as np
G=json.load(open('/work/pass_9/games.json')); B=np.load('/work/pass_9/boards.npy')
def show(b):
    ch={1:'X',-1:'O',0:'.'}
    return [''.join(ch[int(b[r*9+c])] for c in range(9)) for r in range(9)]
for gi in [33,48]:
    g=G[gi]; n=len(g['moves'])
    rows=[show(B[gi,j]) for j in range(n-6,n)]
    print(gi,g['net_color'],[(m['by'][0], divmod(m['move'],9) if m['move']<81 else 'P', [(divmod(a,9) if a<81 else 'P',round(p,3)) for a,p in m.get('top5',[])[:4]]) for m in g['moves'][n-6:n]])
    for r in range(9): print('   '+'   '.join(x[r] for x in rows))
```