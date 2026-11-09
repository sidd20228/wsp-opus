Same 96.1% with higher margin; I'll keep λ=3 (v26). Let me inspect its 5 losses for anything systematic.

```python
import json, numpy as np
G=json.load(open('/work/pass_30/games.json')); B=np.load('/work/pass_30/boards.npy')
def show(b):
    ch={1:'X',-1:'O',0:'.'}
    return [''.join(ch[int(b[r*9+c])] for c in range(9)) for r in range(9)]
for gi,g in enumerate(G):
    if g['result']==-1:
        n=len(g['moves'])
        mine=1 if g['net_color']=='black' else -1
        big=[(k,int(((B[gi,k-1]==mine)&(B[gi,k]==0)).sum())) for k in range(1,n)]
        big=[x for x in big if x[1]>=3]
        print(gi,g['net_color'],g['end'],g['margin'],'len',n,'our losses>=3:',big)
        for r in range(9): print('    '+show(B[gi,n-1])[r])
```