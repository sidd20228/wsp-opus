Both repetition losses are the same double-ko cycle (same game index in two passes — game seeds appear fixed, so passes are correlated). Let me measure how often my net takes kos at all, to judge the cost of discouraging ko recaptures.

```python
exec(open('/work/go.py').read())
import json, numpy as np
from collections import Counter
def tob(arr): return [1 if x==1 else (2 if x==-1 else 0) for x in arr]
c=Counter()
for pk in [25,26,27]:
    G=json.load(open('/work/pass_%d/games.json'%pk)); Bd=np.load('/work/pass_%d/boards.npy'%pk)
    for gi,g in enumerate(G):
        netc=1 if g['net_color']=='black' else 2
        kos=0; late=0
        for k,m in enumerate(g['moves']):
            if m['by']!='net' or m['move']==81 or k==0: continue
            prev=tob(Bd[gi,k-1]); cur=tob(Bd[gi,k]); p=m['move']
            capt=[q for q in range(81) if prev[q]==3-netc and cur[q]==0]
            if len(capt)==1:
                gg,l=group(cur,p)
                if len(gg)==1 and len(l)==1:
                    kos+=1; late+= (k>=90)
        c['games']+=1; c['kos']+=kos; c['late_kos']+=late; c['games_with_ko']+=(kos>0)
print(c)
```