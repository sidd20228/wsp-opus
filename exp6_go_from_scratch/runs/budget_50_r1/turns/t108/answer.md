θ=2.5 gives the same overall 94.5% (Black up, White down). I'll keep v20 (θ=2.0) as the base. Next probe: stronger territorial weight α 0.5 → 0.7.

```python
exec(open('/work/build8.py').read())
import json
cfg=json.load(open('/work/cfg_v20.json')); cfg.update(alpha=0.7)
arrs,layout=build8(cfg,inp)
for k,v in arrs.items(): save(k,v)
json.dump(cfg,open('/work/cfg_v22.json','w'))
print(cfg)
```