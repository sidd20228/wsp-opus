Everything is numerically sane and finite. I have passes to spare, so let me re-test one principled idea in the current configuration: discounting the influence of weak (≤2-liberty) stones, at a milder strength (ω=0.5).

```python
exec(open('/work/build8.py').read())
import json
cfg=json.load(open('/work/cfg_v29.json')); cfg.update(omega=0.5)
arrs,layout=build8(cfg,inp)
for k,v in arrs.items(): save(k,v)
json.dump(cfg,open('/work/cfg_v30.json','w'))
print(cfg)
```