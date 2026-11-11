Strict own-territory fills are rare (0.7 per game), so the checkerboard forms in regions that are still technically contested. Let me watch how the bottom edge shape formed in game 11.

```python
import json, numpy as np
d='/work/pass_14/'
G=json.load(open(d+'games.json')); B=np.load(d+'boards.npy')
def bs(b):
    m={0:'.',1:'X',-1:'O'}
    return [''.join(m[int(b[r*9+c])] for c in range(9)) for r in range(9)]
gi=11; mv=G[gi]['moves']
for k in [35,45,55,65]:
    print('ply',k); print('\n'.join(bs(B[gi,k])))
print([(m['ply'],m['by'][0],divmod(m['move'],9) if m['move']<81 else 'P',m['top5'][0][1] if m.get('top5') else '') for m in mv[36:76]])
```