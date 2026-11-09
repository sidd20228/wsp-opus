A 10-ply double-fight cycle — rare and hard to prevent in general. I'll keep v19 (phase-faded line bias: higher margin, more blowouts). Next variant: broader "contested" band (θ 1.5 → 2.0).

```python
exec(open('/work/build8.py').read())
import json
cfg=json.load(open('/work/cfg_v19.json')); cfg.update(theta=2.0)
arrs,layout=build8(cfg,inp)
for k,v in arrs.items(): save(k,v)
json.dump(cfg,open('/work/cfg_v20.json','w'))
print(cfg)
```