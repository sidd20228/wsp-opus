v12 isn't better; I'll revert to v11. Game 115 lost 109 stones — let me check whether that's a late-game self-inflicted collapse (e.g., filling own eye space) or a real fight.

```python
import json, numpy as np
G=json.load(open('/work/pass_15/games.json')); B=np.load('/work/pass_15/boards.npy')
def show(b):
    ch={1:'X',-1:'O',0:'.'}
    return [''.join(ch[int(b[r*9+c])] for c in range(9)) for r in range(9)]
gi=115; g=G[gi]; n=len(g['moves']); print(g['net_color'], n)
lost=[]
prev=None
for k in range(n):
    b=B[gi,k]
    if prev is not None:
        mine=-1 if g['net_color']=='white' else 1
        l=int(((prev==mine)&(b==0)).sum())
        if l>0: lost.append((k,l,g['moves'][k]['by']))
    prev=b
print(lost)
for k0 in [70,85,100]:
    rows=[show(B[gi,j]) for j in range(k0,k0+8)]
    print(' '.join('%s%s'%(g['moves'][j]['by'][0], divmod(g['moves'][j]['move'],9) if g['moves'][j]['move']<81 else 'P') for j in range(k0,k0+8)))
    for r in range(9): print('  '+' '.join(x[r] for x in rows))
```