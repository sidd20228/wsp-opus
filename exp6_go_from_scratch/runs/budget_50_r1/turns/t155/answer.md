The cycle starts with a 2-stone self-atari lured by a false capture of a huge 2-liberty group (+240), outweighing the self-atari-connect penalty (360) — and with the board nearly full, it still beat passing (−180). I'll raise κ (6→10) so such moves fall below the pass threshold.

```python
exec(open('/work/build8.py').read())
import json
cfg=json.load(open('/work/cfg_v33.json')); cfg.update(kap=10.0)
arrs,layout=build8(cfg,inp)
for k,v in arrs.items(): save(k,v)
json.dump(cfg,open('/work/cfg_v35.json','w'))
print(cfg)
```