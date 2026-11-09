Fewer close games and a better close-game margin. Let me continue the trend with α=0.9.

```python
exec(open('/work/build8.py').read())
import json
cfg=json.load(open('/work/cfg_v20.json')); cfg.update(alpha=0.9)
arrs,layout=build8(cfg,inp)
for k,v in arrs.items(): save(k,v)
json.dump(cfg,open('/work/cfg_v23.json','w'))
print(cfg)
```