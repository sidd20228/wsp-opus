Many losses are by 0.5–1.5 points, so endgame precision matters. Let me trace how the score evolves over the last part of a few close losses (estimating area score after each ply with Go-rule code).

```python
exec(open('/work/go.py').read())
import json, numpy as np
G=json.load(open('/work/pass_19/games.json')); B=np.load('/work/pass_19/boards.npy')
def tob(arr): return [1 if x==1 else (2 if x==-1 else 0) for x in arr]
def show(b):
    ch={1:'X',-1:'O',0:'.'}
    return [''.join(ch[int(b[r*9+c])] for c in range(9)) for r in range(9)]
for gi in [4,15,57]:
    g=G[gi]; n=len(g['moves'])
    sc=[score(tob(B[gi,k])) for k in range(n)]
    print(gi,g['net_color'],'final',g['margin'])
    print('  score(B-W-komi) by ply from 40:',[ (k,round(sc[k],1)) for k in range(40,n,4)])
    k0=max(40,n-40)
    rows=[show(B[gi,j]) for j in range(k0,n,5)]
    print('  moves',[(m['by'][0], divmod(m['move'],9) if m['move']<81 else 'P') for m in g['moves'][k0:]])
    for r in range(9): print('   '+' '.join(x[r] for x in rows))
```