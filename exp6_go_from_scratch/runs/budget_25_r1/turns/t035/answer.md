The net still passes with probability near 1.0 in the double-ko positions, so every other legal move must score below −16. Let me look at the board at one of those passes.

```python
import json, numpy as np
d='/work/pass_6/'
G=json.load(open(d+'games.json')); B=np.load(d+'boards.npy')
def bs(b):
    m={0:'.',1:'X',-1:'O'}
    return [''.join(m[int(b[r*9+c])] for c in range(9)) for r in range(9)]
for gi in [36,42,68]:
    g=G[gi]; mv=g['moves']; n=len(mv); last=B[gi,n-1]
    k=[j for j in range(n-1) if (B[gi,j]==last).all()][-1]
    # pass at ply k+3 -> board before is after ply k+2
    print(gi,g['net_color'],'pass ply',k+3,mv[k+3])
    print('\n'.join(bs(B[gi,k+2])))
```