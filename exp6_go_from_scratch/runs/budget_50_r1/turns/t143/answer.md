Not better (123 vs 125); v29 stays the baseline. Since the session runs until all passes are used, I'll spend the remaining passes on a few more single-change experiments from v29, adopting only clear gains (≥3 net wins on the paired seed set). First: phase-fade length S0 40→60 stones.

```python
exec(open('/work/build8.py').read())
import json
cfg=json.load(open('/work/cfg_v29.json')); cfg.update(S0=60.0)
arrs,layout=build8(cfg,inp)
for k,v in arrs.items(): save(k,v)
json.dump(cfg,open('/work/cfg_v31.json','w'))
print(cfg)
```