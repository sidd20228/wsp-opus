```python
import json, numpy as np
G=json.load(open('/work/pass_3/games.json'))
B=np.load('/work/pass_3/boards.npy'); Vd=np.load('/work/pass_3/valid.npy')
def pb_reg(bb,rows=range(4,9),cols=range(3,9)):
    for r in rows: print(''.join({0:'.',1:'X',-1:'O'}[int(bb[r*9+c])] for c in cols))
gi=37; n=int(Vd[gi].sum())
for ply in range(n-7,n):
    m=G[gi]['moves'][ply]; print('ply',ply,m['by'],divmod(m['move'],9) if m['move']<81 else 'pass'); pb_reg(B[gi][ply])
```