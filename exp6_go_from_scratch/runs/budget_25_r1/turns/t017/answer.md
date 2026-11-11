Before spending a games pass, I'll check the new design on hand-made tests and on the recorded positions that caused cycles or big losses.

```python
import json, numpy as np
d='/work/pass_2/'
G=json.load(open(d+'games.json')); B=np.load(d+'boards.npy')
def bs(b):
    m={0:'.',1:'X',-1:'O'}
    return [''.join(m[int(b[r*9+c])] for c in range(9)) for r in range(9)]
E=['.........']*9
pos=[]
pos.append({"board":E,"to_move":"black"})
pos.append({"board":[".........",".........",".........","....X....","...XO....","....X....",".........",".........","........."],"to_move":"black"})
pos.append({"board":[".........",".........",".........","....O....","...OX....","....O....",".........",".........","........."],"to_move":"black"})
pos.append({"board":[".........",".........",".........","XXXXXXXX.","OOOOOOOO.","XXXXXXXX.",".........",".........","........."],"to_move":"black"})
for gi,ply in [(69,98),(51,87),(62,95),(123,80),(59,71),(92,78),(36,109)]:
    g=G[gi]
    pos.append({"board":bs(B[gi,ply]),"to_move":g['net_color']})
pos.append({"board":[".X.X.....","XXXX.....",".........",".........","....O....",".........",".........","......O..","........."],"to_move":"black"})
json.dump(pos,open('/work/test_pos3.json','w'))
print(json.dumps({"positions":pos,"activations":True}))
```