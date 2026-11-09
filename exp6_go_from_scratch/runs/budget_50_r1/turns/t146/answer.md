Paired results are even (3–3, margin +3±2.5) — not significant, so I keep v29. Next experiment: weaker own-sphere penalty (β_own 2→1), which would allow more solid defensive moves.

```python
exec(open('/work/build8.py').read())
import json
cfg=json.load(open('/work/cfg_v29.json')); cfg.update(beta_own=1.0)
arrs,layout=build8(cfg,inp)
for k,v in arrs.items(): save(k,v)
json.dump(cfg,open('/work/cfg_v32.json','w'))
print(cfg)
```