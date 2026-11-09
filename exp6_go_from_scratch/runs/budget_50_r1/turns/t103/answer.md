```python
import json, numpy as np
for pk,name in [(17,'v14'),(22,'v19'),(23,'v20 th2')]:
    G=json.load(open('/work/pass_%d/games.json'%pk))
    mb=np.array([g['margin'] for g in G if g['net_color']=='black'])
    mw=np.array([g['margin'] for g in G if g['net_color']=='white'])
    cb=mb[np.abs(mb)<40]
    print(pk,name,'B close n=%d mean=%.2f'%(len(cb),cb.mean()),' Bwin=%.3f Wwin=%.3f'%((mb>0).mean(),(mw>0).mean()), 'blowouts B',(mb>=40).sum(),'W',(mw>=40).sum(), 'median B %.1f W %.1f'%(np.median(mb),np.median(mw)))
    print('   losses',[(g['net_color'],g['margin'],g['end']) for g in G if g['result']==-1])
```