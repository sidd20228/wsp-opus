```python
import json, numpy as np
G=json.load(open('/work/pass_3/games.json'))
B=np.load('/work/pass_3/boards.npy')
def pb(bb):
    for r in range(9): print(''.join({0:'.',1:'X',-1:'O'}[int(v)] for v in bb[r*9:r*9+9]))
gi=42
for ply in (60,66,70,72,74,76):
    m=G[gi]['moves'][ply]
    print('ply',ply,m['by'],divmod(m['move'],9) if m['move']<81 else 'P',[(divmod(a,9) if a<81 else 'P',p) for a,p in m.get('top5',[])[:3]])
    pb(B[gi][ply])
```