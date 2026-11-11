The final boards show close territory splits, and my net starts passing around ply 70–84 while the opponent keeps playing. In v2 it passed about 2 times per game; now about 8. Let me see what the opponent gains during those passes.

```python
import json, numpy as np
d='/work/pass_7/'
G=json.load(open(d+'games.json')); B=np.load(d+'boards.npy')
def bs(b):
    m={0:'.',1:'X',-1:'O'}
    return [''.join(m[int(b[r*9+c])] for c in range(9)) for r in range(9)]
gi=32; g=G[gi]
print('\n'.join(bs(B[gi,79])))
for m in g['moves'][80:102]:
    print(m['ply'],m['by'],divmod(m['move'],9) if m['move']<81 else 'pass', m.get('top5','')[:3] if m.get('top5') else '')
```