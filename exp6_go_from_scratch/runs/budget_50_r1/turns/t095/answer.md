Black loses 28/256 (all close, by two passes); White loses 1/256. Let me look at Black's margin distribution to see how many games are near the komi threshold.

```python
import json, numpy as np
m=[]
for pk in [13,14,17,19]:
    G=json.load(open('/work/pass_%d/games.json'%pk))
    m+= [g['margin'] for g in G if g['net_color']=='black']
m=np.array(m); print(len(m))
print(np.histogram(m,bins=[-100,-20,-10,-5,0,5,10,20,40,80,200]))
mw=[]
for pk in [13,14,17,19]:
    G=json.load(open('/work/pass_%d/games.json'%pk))
    mw+= [g['margin'] for g in G if g['net_color']=='white']
print(np.histogram(np.array(mw),bins=[-100,-20,-10,-5,0,5,10,20,40,80,200]))
```