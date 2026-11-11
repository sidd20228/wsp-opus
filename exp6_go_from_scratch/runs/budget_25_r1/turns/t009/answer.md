The pattern is clear. The net makes a self-atari, often because the parity code falsely flags an opponent 2-liberty chain as capturable. The opponent then captures two stones, and my forced recapture recreates an earlier position, which loses. Let me check this across all repetition games.

```python
import json, numpy as np
d='/work/pass_2/'
G=json.load(open(d+'games.json')); B=np.load(d+'boards.npy')
reps=[i for i,g in enumerate(G) if 'rep' in g['end']]
for gi in reps:
    g=G[gi]; mv=g['moves']; n=len(mv)
    last=B[gi,n-1]
    match=[k for k in range(n-1) if (B[gi,k]==last).all()]
    k=match[-1] if match else None
    seq=[(m['by'][0],m['move'],(m['top5'][0][1] if m.get('top5') else None)) for m in mv[k+1:]] if k is not None else None
    print(gi,g['opponent'][:2],g['net_color'][0],'len',n,'margin',g['margin'],'repeats ply',k,'seq',seq)
```