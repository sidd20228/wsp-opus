Let me quantify wasted moves: plays inside an empty region bordered only by the mover's own stones (zero gain under area scoring), counted for both sides before the endgame.

```python
exec(open('/work/go.py').read())
import json, numpy as np
from collections import Counter
def tob(arr): return [1 if x==1 else (2 if x==-1 else 0) for x in arr]
def region_owner(b,p):
    reg={p}; st=[p]; border=set()
    while st:
        x=st.pop()
        for y in NB[x]:
            if b[y]==0 and y not in reg: reg.add(y); st.append(y)
            elif b[y]: border.add(b[y])
    return border, len(reg)
tot=Counter()
for pk in [17,19]:
    G=json.load(open('/work/pass_%d/games.json'%pk)); Bd=np.load('/work/pass_%d/boards.npy'%pk)
    for gi,g in enumerate(G):
        netc=1 if g['net_color']=='black' else 2
        for k,m in enumerate(g['moves']):
            if m['move']==81 or k==0: continue
            if k>70: break
            prev=tob(Bd[gi,k-1]); col=netc if m['by']=='net' else 3-netc
            border,sz=region_owner(prev,m['move'])
            key=(m['by'], 'black' if col==1 else 'white')
            tot[key,'moves']+=1
            if border=={col}: tot[key,'own_region']+=1
            if border=={3-col}: tot[key,'opp_region']+=1
for k in sorted(tot): print(k, tot[k])
```