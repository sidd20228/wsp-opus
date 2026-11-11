The strong opponent plays mostly on the second line (59% early, 68% midgame), while my net plays 3rd–4th line and center. In the midgame the opponent also connects solidly more often. Let me watch one lost game every ten plies to see how territory slips away.

```python
import json, numpy as np
d='/work/pass_7/'
G=json.load(open(d+'games.json')); B=np.load(d+'boards.npy')
def bs(b):
    m={0:'.',1:'X',-1:'O'}
    return [''.join(m[int(b[r*9+c])] for c in range(9)) for r in range(9)]
gi=33; g=G[gi]
print(g['net_color'])
for k in [9,19,29,39,49]:
    rows=bs(B[gi,k])
    print('ply',k,'  next net move', [ (divmod(m['move'],9) if m['move']<81 else 'pass', m['top5'][:3]) for m in g['moves'][k+1:k+2]])
    print('\n'.join(rows))
```