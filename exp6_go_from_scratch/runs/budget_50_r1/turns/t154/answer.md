α=1.2 is worse (123); α≈1.0 is the sweet spot. Restoring v33, and looking at the weak-game 3-cycle in the same run.

```python
exec(open('/work/build8.py').read())
import json, numpy as np
cfg=json.load(open('/work/cfg_v33.json'))
arrs,layout=build8(cfg,inp)
for k,v in arrs.items(): save(k,v)
print('restored v33', cfg['alpha'])
G=json.load(open('/work/pass_41/games.json')); B=np.load('/work/pass_41/boards.npy')
def show(b):
    ch={1:'X',-1:'O',0:'.'}
    return [''.join(ch[int(b[r*9+c])] for c in range(9)) for r in range(9)]
gi=70; g=G[gi]; n=len(g['moves'])
rows=[show(B[gi,j]) for j in range(n-4,n)]
for r in range(9): print('   '+'  '.join(x[r] for x in rows))
m=g['moves'][n-3]; print(m['by'],divmod(m['move'],9),m['top5'])
```