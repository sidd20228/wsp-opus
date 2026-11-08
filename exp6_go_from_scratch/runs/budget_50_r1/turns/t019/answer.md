Losses without repetition are territorial: the opponent simply gets more area, while my net wastes moves filling its own area. Let me watch one game's progression.

```python
import json, numpy as np
G=json.load(open('/work/pass_2/games.json')); B=np.load('/work/pass_2/boards.npy')
def show(b):
    ch={1:'X',-1:'O',0:'.'}
    return [''.join(ch[int(b[r*9+c])] for c in range(9)) for r in range(9)]
gi=55; g=G[gi]
print(g['net_color'], g['margin'])
for k in range(0,60,6):
    rows=[show(B[gi,j]) for j in range(k,k+6)]
    mvs=[(g['moves'][j]['by'][0], divmod(g['moves'][j]['move'],9) if g['moves'][j]['move']<81 else 'P', [ (divmod(a,9) if a<81 else 'P',round(pp,2)) for a,pp in g['moves'][j].get('top5',[])[:3]]) for j in range(k,k+6)]
    for m in mvs: print(m)
    for r in range(9): print('   '.join(rw[r] for rw in rows))
```