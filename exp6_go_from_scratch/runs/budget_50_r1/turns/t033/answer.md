My net loses the territory battle: scattered stones, while the opponent builds solid walls. Let me look at the net's choices move by move in one game.

```python
import json, numpy as np
G=json.load(open('/work/pass_5/games.json')); B=np.load('/work/pass_5/boards.npy')
def show(b):
    ch={1:'X',-1:'O',0:'.'}
    return [''.join(ch[int(b[r*9+c])] for c in range(9)) for r in range(9)]
gi=35; g=G[gi]
for k in range(0,36,6):
    rows=[show(B[gi,j]) for j in range(k,k+6)]
    for j in range(k,k+6):
        m=g['moves'][j]
        print(m['by'][0], divmod(m['move'],9) if m['move']<81 else 'P', [(divmod(a,9) if a<81 else 'P',round(p,2)) for a,p in m.get('top5',[])])
    for r in range(9): print('   '+'   '.join(x[r] for x in rows))
```