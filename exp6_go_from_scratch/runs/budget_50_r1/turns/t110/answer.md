98.4% (126/128) with α=0.7 — a notable jump. Let me look at its metrics, then confirm since some of it may be luck.

```python
import json, numpy as np
for pk,name in [(23,'v20 a.5'),(25,'v22 a.7')]:
    G=json.load(open('/work/pass_%d/games.json'%pk))
    mb=np.array([g['margin'] for g in G if g['net_color']=='black'])
    mw=np.array([g['margin'] for g in G if g['net_color']=='white'])
    cb=mb[np.abs(mb)<40]
    print(pk,name,'B close n=%d mean=%.2f'%(len(cb),cb.mean()),' Bwin=%.3f Wwin=%.3f'%((mb>0).mean(),(mw>0).mean()), 'blowouts B',(mb>=40).sum(),'W',(mw>=40).sum())
    print('   losses',[(g['net_color'],g['margin'],g['end']) for g in G if g['result']==-1])
```