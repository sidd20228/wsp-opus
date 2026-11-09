Not better (92.2%); v14 remains the best (pooled 363/384 = 94.5%). Let me examine endgame move choices in a close Black loss for obvious misorderings.

```python
import json, numpy as np
exec(open('/work/go.py').read())
G=json.load(open('/work/pass_17/games.json')); B=np.load('/work/pass_17/boards.npy')
def tob(arr): return [1 if x==1 else (2 if x==-1 else 0) for x in arr]
def show(b):
    ch={1:'X',-1:'O',0:'.'}
    return [''.join(ch[int(b[r*9+c])] for c in range(9)) for r in range(9)]
gi=28; g=G[gi]; n=len(g['moves']); print(g['net_color'], g['margin'], n)
for k in range(44,n-1,2):
    m=g['moves'][k]
    if m['by']!='net': continue
    print(k, 'net', divmod(m['move'],9) if m['move']<81 else 'P', [(divmod(a,9) if a<81 else 'P',round(p,2)) for a,p in m['top5'][:4]], '| opp', divmod(g['moves'][k+1]['move'],9) if g['moves'][k+1]['move']<81 else 'P', ' score', score(tob(B[gi,k+1])))
for r in range(9): print('   '+'  '.join(show(B[gi,j])[r] for j in (44,58,72)))
```