Random 128/128. The strong benchmark is now saturated (127/128), so further tuning can't be measured on it — only risked. Let me verify the saved tensors exactly match the v35 build.

```python
exec(open('/work/build8.py').read())
import json
cfg=json.load(open('/work/cfg_v35.json'))
arrs,layout=build8(cfg,inp)
ok=True
for k,v in arrs.items():
    s=load(k); d=float(np.abs(s-v).max()); ok&= (d==0.0)
    print(k, s.shape, 'maxdiff', d)
print('ALL MATCH' if ok else 'MISMATCH', sorted(saved()))
json.dump({'final':'v35','cfg':cfg,'layout':layout},open('/work/FINAL.json','w'))
```