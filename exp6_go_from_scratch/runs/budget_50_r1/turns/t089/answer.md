Our wasted-move rate is only slightly above the opponent's. The biggest past gain came from the 2-liberty tactics, so let me test stronger weights on them (atari-move 6→12, defend-own-2-lib 8→12).

```python
exec(open('/work/build6.py').read())
import json
cfg=json.load(open('/work/cfg_v14.json')); cfg.update(w_at=12.0, w_esc=12.0)
arrs,layout=build6(cfg,inp)
for k,v in arrs.items(): save(k,v)
json.dump(cfg,open('/work/cfg_v17.json','w'))
print(cfg)
```