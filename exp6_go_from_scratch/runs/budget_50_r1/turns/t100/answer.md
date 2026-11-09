More Black blowouts (50 vs 37) — promising, though overall is within noise. There's a repetition loss while +17.5 ahead; let me inspect it.

```python
import json, numpy as np
G=json.load(open('/work/pass_22/games.json')); B=np.load('/work/pass_22/boards.npy')
def show(b):
    ch={1:'X',-1:'O',0:'.'}
    return [''.join(ch[int(b[r*9+c])] for c in range(9)) for r in range(9)]
for gi,g in enumerate(G):
    if g['end']=='repetition' and g['result']==-1:
        n=len(g['moves']); last=B[gi,n-1]; k0=[k for k in range(n-1) if np.array_equal(B[gi,k],last)][-1]
        print(gi,g['net_color'],'cycle len',n-1-k0)
        for m in g['moves'][k0+1:n]: print('   ',m['by'], divmod(m['move'],9) if m['move']<81 else 'P', [(divmod(a,9) if a<81 else 'P',round(p,3)) for a,p in m.get('top5',[])[:4]])
        rows=[show(B[gi,j]) for j in range(k0,n)]
        for r in range(9): print('   '+'  '.join(x[r] for x in rows))
```