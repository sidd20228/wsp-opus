Repetitions fell from 25 to 8, but my net now captures only 21 stones per game against strong, versus 47 in v2. The KO unit is to blame: "all neighbours are opponent stones" also describes killing a whole group by filling its last eye. Let me check the score losses to see what else is going wrong.

```python
import json, numpy as np
d='/work/pass_7/'
G=json.load(open(d+'games.json')); B=np.load(d+'boards.npy')
def bs(b):
    m={0:'.',1:'X',-1:'O'}
    return [''.join(m[int(b[r*9+c])] for c in range(9)) for r in range(9)]
L=[gi for gi,g in enumerate(G) if g['opponent']=='strong' and g['result']<0 and 'rep' not in g['end']]
print(len(L))
for gi in L[:12]:
    g=G[gi]; n=len(g['moves'])
    npass=[m['ply'] for m in g['moves'] if m['by']=='net' and m['move']==81]
    print(gi,g['net_color'][0],'mg',g['margin'],'len',n,'cap',g['stones_captured_by_net'],'lost',g['stones_lost_by_net'],'net passes at',npass[:6])
for gi in L[:3]:
    n=len(G[gi]['moves']); print(gi,G[gi]['net_color']); print('\n'.join(bs(B[gi,n-1])))
```