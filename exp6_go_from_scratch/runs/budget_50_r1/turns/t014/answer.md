Let me find exactly which earlier position was repeated and the cycle in between.

```python
import json, numpy as np
G=json.load(open('/work/pass_2/games.json')); B=np.load('/work/pass_2/boards.npy'); V=np.load('/work/pass_2/valid.npy')
def show(b):
    ch={1:'X',-1:'O',0:'.'}
    return [''.join(ch[int(b[r*9+c])] for c in range(9)) for r in range(9)]
for gi,g in enumerate(G):
    if g['end']!='repetition': continue
    n=len(g['moves']); last=B[gi,n-1]
    prev=[k for k in range(n-1) if np.array_equal(B[gi,k],last)]
    k0=prev[-1]
    seq=[(m['by'][0], divmod(m['move'],9) if m['move']<81 else 'P') for m in g['moves'][k0+1:n]]
    print(gi,g['opponent'],'cycle from ply',k0,'len',n-1-k0, seq)
```