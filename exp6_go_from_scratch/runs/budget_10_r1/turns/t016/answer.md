```python
import json, numpy as np
G=json.load(open('/work/pass_2/games.json'))
B=np.load('/work/pass_2/boards.npy')
def pb(bb):
    for r in range(9): print(''.join({0:'.',1:'X',-1:'O'}[int(v)] for v in bb[r*9:r*9+9]))
g=G[33]
for ply in (20,40,57,77):
    print('ply',ply); pb(B[33][ply])
for m in g['moves'][50:80]:
    if m['by']=='net': print(m['ply'],divmod(m['move'],9) if m['move']<81 else 'pass',[(divmod(a,9) if a<81 else 'P',p) for a,p in m['top5'][:4]])
```