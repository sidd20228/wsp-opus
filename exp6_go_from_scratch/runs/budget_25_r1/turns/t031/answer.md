Discounting by the adjacent stone's connectivity works:
- A neighbour with no same-colour neighbours (a lone stone) gives no fakes in 6,716 cases.
- With one, two or three same-colour neighbours, about 66%, 41% and 33% of claims are true.

I'll scale the capture bonus by connectivity: full for a lone stone, about 0.4 for a stone with two neighbours in its chain. Before rebuilding, let me check why the weak-opponent result dropped.

```python
import json, numpy as np
d='/work/pass_5/'
G=json.load(open(d+'games.json')); B=np.load(d+'boards.npy')
def bs(b):
    m={0:'.',1:'X',-1:'O'}
    return [''.join(m[int(b[r*9+c])] for c in range(9)) for r in range(9)]
for gi,g in enumerate(G):
    if g['opponent']=='weak' and g['result']<0:
        n=len(g['moves'])
        npass=sum(1 for m in g['moves'] if m['by']=='net' and m['move']==81)
        first_pass=[m['ply'] for m in g['moves'] if m['by']=='net' and m['move']==81][:3]
        print(gi,g['net_color'],g['end'],'margin',g['margin'],'len',n,'cap',g['stones_captured_by_net'],'lost',g['stones_lost_by_net'],'netpasses',npass,first_pass)
gi=[i for i,g in enumerate(G) if g['opponent']=='weak' and g['result']<0 and 'two' in g['end']][0]
n=len(G[gi]['moves'])
for k in [n//2, n-1]:
    print('ply',k); print('\n'.join(bs(B[gi,k])))
```