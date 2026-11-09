Both α=0.7 (126/128) and α=0.9 (124/128) beat α=0.5. I'll settle in between at α=0.8 and verify.

```python
exec(open('/work/build8.py').read())
import json
cfg=json.load(open('/work/cfg_v20.json')); cfg.update(alpha=0.8)
arrs,layout=build8(cfg,inp)
for k,v in arrs.items(): save(k,v)
json.dump(cfg,open('/work/cfg_v24.json','w'))
print(cfg)
```