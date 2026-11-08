The 3-cycles now come from deliberate moves (prob 1.0). Let me look at the board before such a move.

```python
import json, numpy as np
G=json.load(open('/work/pass_5/games.json')); B=np.load('/work/pass_5/boards.npy')
def show(b):
    ch={1:'X',-1:'O',0:'.'}
    return [''.join(ch[int(b[r*9+c])] for c in range(9)) for r in range(9)]
for gi in [42,43,72,0]:
    g=G[gi]; n=len(g['moves']); last=B[gi,n-1]
    k0=[k for k in range(n-1) if np.array_equal(B[gi,k],last)][-1]
    print('game',gi,'net',g['net_color'])
    rows=[show(B[gi,j]) for j in range(k0,n)]
    for r in range(9): print('   '+'   '.join(x[r] for x in rows))
```