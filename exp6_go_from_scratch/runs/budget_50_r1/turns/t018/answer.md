Key facts: exact H/V witnesses are nearly perfect for ≥2 liberties; uncorrected "star" witnesses err ~4–7%. Before restructuring, let me look at the non-repetition losses (final boards) to see what else goes wrong.

```python
import json, numpy as np
G=json.load(open('/work/pass_2/games.json')); B=np.load('/work/pass_2/boards.npy')
def show(b):
    ch={1:'X',-1:'O',0:'.'}
    return [''.join(ch[int(b[r*9+c])] for c in range(9)) for r in range(9)]
k=0
for gi,g in enumerate(G):
    if g['end']=='two passes' and g['result']==-1 and k<5:
        k+=1; n=len(g['moves'])
        print('game',gi,g['opponent'],'net',g['net_color'],'margin',g['margin'],'len',n)
        seq=[(m['by'][0], m['move']) for m in g['moves'][-16:]]
        print(seq)
        rows=show(B[gi,n-1]); 
        for r in rows: print('   ',r)
```