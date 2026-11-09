100% vs weak and random. Continuing to tune against strong: a wider territory-gain kernel (more weight on contested points 2–3 steps away).

```python
exec(open('/work/build8.py').read())
import json
cfg=json.load(open('/work/cfg_v24.json')); cfg.update(conv={0:1.0,1:1.0,2:0.75,3:0.5})
arrs,layout=build8(cfg,inp)
for k,v in arrs.items(): save(k,v)
json.dump(cfg,open('/work/cfg_v25.json','w'))
print(cfg)
```