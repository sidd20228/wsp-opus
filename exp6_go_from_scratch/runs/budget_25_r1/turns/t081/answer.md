Before spending a games pass, I'll check the new KO and eye logic on the exact positions that motivated them: the game-3 connection, the game-15 snapback, the 32-stone eye-fill, and a few cycle starts.

```python
import json, numpy as np
def bs(b):
    m={0:'.',1:'X',-1:'O'}
    return [''.join(m[int(b[r*9+c])] for c in range(9)) for r in range(9)]
pos=[]; meta=[]
def add(pk,gi,ply,note):
    d=f'/work/pass_{pk}/'
    G=json.load(open(d+'games.json')); B=np.load(d+'boards.npy')
    pos.append({"board":bs(B[gi,ply]),"to_move":G[gi]['net_color']}); meta.append((pk,gi,ply,note))
add(15,3,85,'want (4,0) connect if certified')
add(15,15,73,'avoid snapback (0,8)')
add(12,2,101,'avoid (2,0) eye fill 32 stones')
# cycle starts in pass 13 / 14
for pk in [13,14]:
    d=f'/work/pass_{pk}/'
    G=json.load(open(d+'games.json')); B=np.load(d+'boards.npy')
    for gi,g in enumerate(G):
        if 'rep' not in g['end']: continue
        mv=g['moves']; n=len(mv); last=B[gi,n-1]
        k=[j for j in range(n-1) if (B[gi,j]==last).all()][-1]
        if mv[k+1]['by']=='net' and n-k-1==3:
            add(pk,gi,k,f"cycle start avoid {divmod(mv[k+1]['move'],9)}")
json.dump({'pos':pos,'meta':meta},open('/work/test16.json','w'))
print(len(pos))
print(json.dumps({"positions":pos,"activations":False}))
```