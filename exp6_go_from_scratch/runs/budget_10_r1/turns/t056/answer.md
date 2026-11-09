```python
import json, numpy as np
exec(open('/work/go.py').read())
G=json.load(open('/work/pass_8/games.json'))
B=np.load('/work/pass_8/boards.npy')
def pb(bb):
    for r in range(9): print(''.join({0:'.',1:'X',-1:'O'}[int(v)] for v in bb[r*9:r*9+9]))
g=G[6]; print(g['net_color'])
cnt=0
for m in g['moves']:
    if m['by']=='net' and m['move']==81 and cnt<3:
        cnt+=1; ply=m['ply']; print('ply',ply,[(divmod(a,9) if a<81 else 'P',p) for a,p in m['top5']])
        pb(B[g['game'] if 'game' in g else 6][ply-1])
```