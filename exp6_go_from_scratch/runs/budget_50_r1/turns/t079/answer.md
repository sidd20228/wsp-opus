Same 94.5% (Black 89%, White 100%). I'll keep v14. Black's small-margin losses are the remaining gap; let me look at one closely for recurring inefficiencies.

```python
import json, numpy as np
G=json.load(open('/work/pass_17/games.json')); B=np.load('/work/pass_17/boards.npy')
def show(b):
    ch={1:'X',-1:'O',0:'.'}
    return [''.join(ch[int(b[r*9+c])] for c in range(9)) for r in range(9)]
L=[gi for gi,g in enumerate(G) if g['result']==-1]
print([(gi,G[gi]['margin']) for gi in L])
gi=L[0]; g=G[gi]
for k in range(0,60,10):
    js=list(range(k,k+10))
    rows=[show(B[gi,j]) for j in js]
    print(' '.join('%s%s'%(g['moves'][j]['by'][0], divmod(g['moves'][j]['move'],9) if g['moves'][j]['move']<81 else 'P') for j in js))
    print('   ',[(j,[(divmod(a,9) if a<81 else 'P',round(p,2)) for a,p in g['moves'][j]['top5'][:3]]) for j in js if g['moves'][j]['by']=='net'])
    for r in range(9): print('  '+' '.join(x[r] for x in rows))
```