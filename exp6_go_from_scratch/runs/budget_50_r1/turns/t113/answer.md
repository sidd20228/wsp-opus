```python
import json, numpy as np
for pk,name in [(23,'v20 a.5'),(25,'v22 a.7'),(26,'v23 a.9')]:
    G=json.load(open('/work/pass_%d/games.json'%pk))
    mb=np.array([g['margin'] for g in G if g['net_color']=='black'])
    mw=np.array([g['margin'] for g in G if g['net_color']=='white'])
    cb=mb[np.abs(mb)<40]; cw=mw[np.abs(mw)<40]
    print(pk,name,'B close n=%d mean=%.2f'%(len(cb),cb.mean()),'W close n=%d mean=%.2f'%(len(cw),cw.mean() if len(cw) else 0),' Bwin=%.3f Wwin=%.3f'%((mb>0).mean(),(mw>0).mean()), 'blowouts B',(mb>=40).sum(),'W',(mw>=40).sum())
    print('   losses',[(g['net_color'],g['margin'],g['end']) for g in G if g['result']==-1])
```