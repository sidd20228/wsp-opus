v36 does not fix position C (the capture there is outweighed by an atari-move elsewhere) and it un-blocks three known bad self-atari moves (positions 3, 5, 6). The two signatures (capture-to-save vs. false-capture self-atari) are indistinguishable to my features, and v35's conservative choice is the one validated at 127/128 / 128/128 / 128/128. Reverting to v35 and re-verifying the saved tensors.

```python
exec(open('/work/build8.py').read())
import json
cfg=json.load(open('/work/cfg_v35.json'))
assert cfg.get('y_mode',0)==0
arrs,layout=build8(cfg,inp)
for k,v in arrs.items(): save(k,v)
ok=all(float(np.abs(load(k)-v).max())==0.0 for k,v in arrs.items())
print('restored v35, verified:',ok, sorted(saved())==sorted(NAMES))
print({k:float(np.abs(v).max()) for k,v in arrs.items()})
```