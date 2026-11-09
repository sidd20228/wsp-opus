v8: 79.7% over 128 games. Now the bump-positional variant with the same eye fix.

```python
exec(open('/work/build4.py').read())
import json
cfg=json.load(open('/work/cfg_v7.json')); cfg['w_eye']=300.0
arrs,layout=build4(cfg,inp)
for k,v in arrs.items(): save(k,v)
json.dump(cfg,open('/work/cfg_v9.json','w')); json.dump(layout,open('/work/layout_v9.json','w'))
print('saved v9', cfg)
```