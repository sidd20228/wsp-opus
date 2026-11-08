Now many losses come from passing too early (first net pass around move 47–70) while the opponent keeps playing. Let me see what happens after our first pass in a lost game.

```python
import json, numpy as np
G=json.load(open('/work/pass_7/games.json')); B=np.load('/work/pass_7/boards.npy')
def show(b):
    ch={1:'X',-1:'O',0:'.'}
    return [''.join(ch[int(b[r*9+c])] for c in range(9)) for r in range(9)]
L=[(gi,g) for gi,g in enumerate(G) if g['result']==-1 and g['end']=='two passes' and g['opponent']=='strong']
print([(gi,g['margin']) for gi,g in L])
for gi,g in L[:3]:
    fp=[m['ply'] for m in g['moves'] if m['by']=='net' and m['move']==81][0]
    n=len(g['moves'])
    print('game',gi,g['net_color'],'margin',g['margin'],'first pass',fp,'len',n)
    rows=[show(B[gi,j]) for j in (fp-1,min(fp+10,n-1),n-1)]
    for r in range(9): print('   '+'   '.join(x[r] for x in rows))
    m=g['moves'][fp]; print('   top5 at first pass',[(divmod(a,9) if a<81 else 'P',round(p,3)) for a,p in m['top5']])
    print('   after:',[(x['by'][0], divmod(x['move'],9) if x['move']<81 else 'P') for x in g['moves'][fp:fp+16]])
```