v28 (+1 net) — keep it. Next, with the self-atari gate in place, retry a stronger atari-move weight (w_at 6→9).

```python
exec(open('/work/build8.py').read())
import json
cfg=json.load(open('/work/cfg_v28.json')); cfg.update(w_at=9.0)
arrs,layout=build8(cfg,inp)
for k,v in arrs.items(): save(k,v)
json.dump(cfg,open('/work/cfg_v29.json','w'))
print(cfg)
```