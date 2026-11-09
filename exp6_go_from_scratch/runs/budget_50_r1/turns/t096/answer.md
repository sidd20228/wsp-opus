When no group dies, Black's territorial margin is centered near +2 — a coin flip against komi. Let me use a lower-noise metric: mean Black margin over non-blowout games (|margin|<40), per pass/variant.

```python
import json, numpy as np
variants={13:'v10',14:'v11',17:'v14',19:'v16(4-4)',15:'v12(omega)',18:'v15(beta4)',20:'v17(tac12)',21:'v18(gnorm)',11:'v9',10:'v8'}
for pk,name in sorted(variants.items()):
    G=json.load(open('/work/pass_%d/games.json'%pk))
    mb=np.array([g['margin'] for g in G if g['net_color']=='black'])
    mw=np.array([g['margin'] for g in G if g['net_color']=='white'])
    cb=mb[np.abs(mb)<40]; cw=mw[np.abs(mw)<40]
    print(pk,name,'B close n=%d mean=%.2f se=%.2f'%(len(cb),cb.mean(),cb.std()/np.sqrt(len(cb))),' W close n=%d mean=%.2f'%(len(cw),cw.mean() if len(cw) else 0), ' Bwin=%.3f'%(mb>0).mean())
```