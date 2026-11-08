The pass logit is the problem: with sharper positional penalties, by move ~50–70 every move looks worse than passing, so the net stops defending while the opponent invades. I'll make passing a true last resort (P0 = −60) and raise the eye-fill penalty so true eyes stay protected.

```python
exec(open('/work/build3.py').read())
import json
cfg=json.load(open('/work/cfg_v5.json'))
cfg.update(P0=-60.0, w_eye=200.0)
arrs,layout=build3(cfg,inp)
for k,v in arrs.items(): save(k,v)
json.dump(cfg,open('/work/cfg_v6.json','w')); json.dump(layout,open('/work/layout_v6.json','w'))
print(cfg)
```