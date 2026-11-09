Six of seven losses are as Black (komi). Let me step through game 15 (−15.5) to look for clear mistakes.

```python
import json, numpy as np
G=json.load(open('/work/pass_13/games.json')); B=np.load('/work/pass_13/boards.npy')
def show(b):
    ch={1:'X',-1:'O',0:'.'}
    return [''.join(ch[int(b[r*9+c])] for c in range(9)) for r in range(9)]
gi=15; g=G[gi]; print(g['net_color'])
for k in range(20,70,10):
    js=list(range(k,k+10))
    rows=[show(B[gi,j]) for j in js]
    print(' '.join('%s%s'%(g['moves'][j]['by'][0], divmod(g['moves'][j]['move'],9) if g['moves'][j]['move']<81 else 'P') for j in js))
    for j in js:
        m=g['moves'][j]
        if m['by']=='net': print('    ',j,[(divmod(a,9) if a<81 else 'P',round(p,2)) for a,p in m['top5'][:4]])
    for r in range(9): print('  '+' '.join(x[r] for x in rows))
```