The center group was already lost in a capturing race (the opponent took the vital point); the gate fix is correct and didn't hurt, so I keep v27. With the bad escape case now gated, let me retry a stronger defence weight (w_esc 8→12).

```python
exec(open('/work/build8.py').read())
import json
cfg=json.load(open('/work/cfg_v27.json')); cfg.update(w_esc=12.0)
arrs,layout=build8(cfg,inp)
for k,v in arrs.items(): save(k,v)
json.dump(cfg,open('/work/cfg_v28.json','w'))
print(cfg)
```