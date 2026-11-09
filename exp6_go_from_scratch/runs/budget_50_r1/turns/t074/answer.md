Random 100%, weak 98.4% (1 loss). Let me check that loss and the repetitions.

```python
import json, numpy as np
G=json.load(open('/work/pass_16/games.json')); B=np.load('/work/pass_16/boards.npy')
def show(b):
    ch={1:'X',-1:'O',0:'.'}
    return [''.join(ch[int(b[r*9+c])] for c in range(9)) for r in range(9)]
for gi,g in enumerate(G):
    if g['result']==-1 or g['end']=='repetition':
        n=len(g['moves'])
        s=f"{gi} {g['opponent']} {g['net_color']} {g['end']} res {g['result']} margin {g['margin']} len {n}"
        if g['end']=='repetition':
            last=B[gi,n-1]; k0=[k for k in range(n-1) if np.array_equal(B[gi,k],last)][-1]
            s+=' cycle '+str([(m['by'][0], divmod(m['move'],9) if m['move']<81 else 'P', [(divmod(a,9) if a<81 else 'P',round(p,2)) for a,p in m.get('top5',[])[:3]]) for m in g['moves'][k0+1:n]][:8])
            rows=[show(B[gi,j]) for j in range(k0,n)]
            print(s)
            for r in range(9): print('   '+'  '.join(x[r] for x in rows))
        else: print(s)
```