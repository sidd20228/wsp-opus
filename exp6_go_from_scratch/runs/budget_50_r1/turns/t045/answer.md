Only one repetition loss left. Let me inspect the close two-pass losses (final boards and the last moves).

```python
import json, numpy as np
G=json.load(open('/work/pass_8/games.json')); B=np.load('/work/pass_8/boards.npy')
exec(open('/work/go.py').read())
def show(b):
    ch={1:'X',-1:'O',0:'.'}
    return [''.join(ch[int(b[r*9+c])] for c in range(9)) for r in range(9)]
for gi in [34,69,84,92,32]:
    g=G[gi]; n=len(g['moves'])
    print('game',gi,g['net_color'],g['margin'])
    rows=[show(B[gi,j]) for j in (n-21,n-11,n-1)]
    for r in range(9): print('   '+'   '.join(x[r] for x in rows))
    print('   last moves',[(m['by'][0], divmod(m['move'],9) if m['move']<81 else 'P', [(divmod(a,9) if a<81 else 'P',round(p,2)) for a,p in m.get('top5',[])[:2]]) for m in g['moves'][-12:]])
```