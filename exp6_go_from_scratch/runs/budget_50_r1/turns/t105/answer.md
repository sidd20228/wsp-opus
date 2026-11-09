Both are double-ko cycles where every alternative was a genuine self-atari — very hard to avoid without game history. I'll adopt v20 (θ=2.0). Let me probe further in that direction (θ=2.5).

```python
exec(open('/work/build8.py').read())
import json
cfg=json.load(open('/work/cfg_v20.json')); cfg.update(theta=2.5)
arrs,layout=build8(cfg,inp)
for k,v in arrs.items(): save(k,v)
json.dump(cfg,open('/work/cfg_v21.json','w'))
print(cfg)
```