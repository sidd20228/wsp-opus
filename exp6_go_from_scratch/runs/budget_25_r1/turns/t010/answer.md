Most cycles follow the same three-move shape: my self-atari, their capture, my recapture. Several first moves had very high probability, so I'll inspect a few of those positions.

```python
import json, numpy as np
d='/work/pass_2/'
G=json.load(open(d+'games.json')); B=np.load(d+'boards.npy')
def bs(b,net_black):
    m={0:'.',1:'X',-1:'O'} 
    return [''.join(m[int(b[r*9+c])] for c in range(9)) for r in range(9)]
for gi,ply in [(62,95),(123,80),(51,87),(92,78),(55,93)]:
    g=G[gi]
    print('game',gi,'net',g['net_color'],'ply',ply,'board before net move', g['moves'][ply+1])
    print('\n'.join(bs(B[gi,ply],1)))
```