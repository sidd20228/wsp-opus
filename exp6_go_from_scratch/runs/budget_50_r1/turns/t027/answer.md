The early passing is a calibration bug: the new convolved influence penalty pushes most mid-game moves below the pass logit (−4). Passing also enabled the 5-ply double-ko cycles (my pass let the opponent retake). I'll make passing a last resort (P0 = −15, only eye-fills/self-ataris/deep territory are below it) and strengthen eye/self-atari penalties accordingly.

```python
exec(open('/work/build2.py').read())
import json
cfg=json.load(open('/work/cfg_v2.json'))
cfg.update(w_eye=50.0, w_sa=20.0, w_cap=30.0, w_esc=8.0, w_at=3.0, P0=-15.0)
arrs,layout=build2(cfg,inp)
for k,v in arrs.items(): save(k,v)
json.dump(cfg,open('/work/cfg_v3.json','w')); json.dump(layout,open('/work/layout_v3.json','w'))
print(cfg)
```