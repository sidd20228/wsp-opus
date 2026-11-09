Three losses involve a 16–17 stone group dying mid-game. Let me look at how that happened in game 103.

```python
import json, numpy as np
G=json.load(open('/work/pass_30/games.json')); B=np.load('/work/pass_30/boards.npy')
def show(b):
    ch={1:'X',-1:'O',0:'.'}
    return [''.join(ch[int(b[r*9+c])] for c in range(9)) for r in range(9)]
gi=103; g=G[gi]; print(g['net_color'])
js=list(range(46,63))
for j in js:
    m=g['moves'][j]
    print(j, m['by'], divmod(m['move'],9) if m['move']<81 else 'P', [(divmod(a,9) if a<81 else 'P',round(p,2)) for a,p in m.get('top5',[])[:4]])
for part in (js[:9], js[9:]):
    rows=[show(B[gi,j]) for j in part]
    for r in range(9): print('  '+' '.join(x[r] for x in rows))
```