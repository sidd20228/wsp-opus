Passes are strongly paired (same seeds): v24 vs this variant differ in only 10 games (6–4), so neither is clearly better; I'll keep v24. Next probe: sharper policy (λ 2→3).

```python
exec(open('/work/build8.py').read())
import json
cfg=json.load(open('/work/cfg_v24.json')); cfg.update(lam=3.0)
arrs,layout=build8(cfg,inp)
for k,v in arrs.items(): save(k,v)
json.dump(cfg,open('/work/cfg_v26.json','w'))
print(cfg)
```