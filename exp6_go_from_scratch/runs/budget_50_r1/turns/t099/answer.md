93.75%, Black 90.6%, mean margin 66.5 (highest so far). Let me compute the close-game metric.

```python
import json, numpy as np
for pk,name in [(17,'v14'),(22,'v19 phase-bias')]:
    G=json.load(open('/work/pass_%d/games.json'%pk))
    mb=np.array([g['margin'] for g in G if g['net_color']=='black'])
    mw=np.array([g['margin'] for g in G if g['net_color']=='white'])
    cb=mb[np.abs(mb)<40]; cw=mw[np.abs(mw)<40]
    print(pk,name,'B close n=%d mean=%.2f'%(len(cb),cb.mean()),' W close n=%d mean=%.2f'%(len(cw),cw.mean()), ' Bwin=%.3f Wwin=%.3f'%((mb>0).mean(),(mw>0).mean()), 'blowouts B',(mb>=40).sum(),'W',(mw>=40).sum())
    G2=[g for g in G if g['result']==-1]; print('   losses',[(g['net_color'],g['margin'],g['end']) for g in G2])
```