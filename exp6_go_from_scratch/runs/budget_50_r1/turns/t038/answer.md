YB was firing (with a 110 penalty) on good connecting moves where the point also touches a strong own group. I'll move YB inside the SA unit so strong-neighbor evidence cancels it.

```python
src=open('/work/build3.py').read()
src=src.replace("""        for q in nb(p): E.W[D.u('F3M',q),u]-=2
        E.W[D.u('Z',p),u]+=-1.0/d; E.b[u]=2+2.0/d
        u=E.u('EY',p); E.W[D.u('EYE',p),u]=cfg['w_eye']; E.W[D.u('YB',p),u]=cfg['w_y']""",
"""        kap=cfg['kap']
        for q in nb(p): E.W[D.u('F3M',q),u]-=(2+8*kap)
        E.W[D.u('YB',p),u]+=kap
        E.W[D.u('Z',p),u]+=-1.0/d; E.b[u]=2+2.0/d
        u=E.u('EY',p); E.W[D.u('EYE',p),u]=cfg['w_eye']""")
open('/work/build3.py','w').write(src)
exec(open('/work/build3.py').read())
import json
cfg=json.load(open('/work/cfg_v4.json')); cfg['kap']=6.0; cfg.pop('w_y',None)
arrs,layout=build3(cfg,inp)
for k,v in arrs.items(): save(k,v)
json.dump(cfg,open('/work/cfg_v5.json','w')); json.dump(layout,open('/work/layout_v5.json','w'))
print(cfg)
```