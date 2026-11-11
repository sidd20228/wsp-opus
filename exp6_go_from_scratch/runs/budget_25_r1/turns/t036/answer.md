The double-ko boards show the root cause. My net had filled its own territory down to single-point eyes, so in a double ko its only alternatives to passing were filling an eye. Avoiding the pass that way would eventually kill my own groups. A cleaner fix is to stop taking kos at all: 17 of the 25 repetition losses began with my ko capture. A legal move whose neighbours are all opponent stones (or edge) must capture, and a single-stone capture of that shape is a ko.

But I also see more on-board score losses against strong than in v2 (20 versus 7), so something in v5/v6 weakened ordinary play. Let me look at those games.

```python
import json, numpy as np
d='/work/pass_6/'
G=json.load(open(d+'games.json')); B=np.load(d+'boards.npy')
def bs(b):
    m={0:'.',1:'X',-1:'O'}
    return [''.join(m[int(b[r*9+c])] for c in range(9)) for r in range(9)]
L=[gi for gi,g in enumerate(G) if g['opponent']=='strong' and g['result']<0 and 'rep' not in g['end']]
for gi in L:
    g=G[gi]; n=len(g['moves'])
    npass=sum(1 for m in g['moves'] if m['by']=='net' and m['move']==81)
    print(gi,g['net_color'][0],g['end'],'mg',g['margin'],'len',n,'cap',g['stones_captured_by_net'],'lost',g['stones_lost_by_net'],'np',npass)
gi=L[0]; n=len(G[gi]['moves'])
for k in [30,60,n-1]:
    print('ply',k); print('\n'.join(bs(B[gi,k])))
```