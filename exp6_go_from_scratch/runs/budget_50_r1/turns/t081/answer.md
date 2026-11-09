No group collapses in the losses — they are purely territorial. I'll try a positional variant that discourages filling inside one's own sphere more strongly and weights contested areas a bit more.

```python
exec(open('/work/build6.py').read())
import json
cfg=json.load(open('/work/cfg_v14.json')); cfg.update(beta_own=4.0, alpha=0.6)
arrs,layout=build6(cfg,inp)
for k,v in arrs.items(): save(k,v)
json.dump(cfg,open('/work/cfg_v15.json','w'))
print(cfg)
```