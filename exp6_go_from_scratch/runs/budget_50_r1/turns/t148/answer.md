124 — not better; keeping v29. Next: push the territory weight further (α 0.8→1.0) on top of v29.

```python
exec(open('/work/build8.py').read())
import json
cfg=json.load(open('/work/cfg_v29.json')); cfg.update(alpha=1.0)
arrs,layout=build8(cfg,inp)
for k,v in arrs.items(): save(k,v)
json.dump(cfg,open('/work/cfg_v33.json','w'))
print(cfg)
```