v8C scores 71% overall, close to most other versions (66–72%); v8B's 76% is likely partly luck. Fine tactical details don't change results measurably at 128 games. The one change with a large observed effect was the line prior (A→B: +14 points against strong). So I'll test pushing further in that direction on v8B, making the 2nd line as attractive as the 3rd.

```python
exec(open('/work/build8.py').read())
PC=dict(t1=2.5,W_o=0.3,prior={0:-2.0,1:1.6,2:1.4,3:0.6,4:0.8},prior_late={0:-1.0,1:0.5,2:0.5,3:0.3,4:0.3})
T=build(PC)
for k,v in T.items(): save(k,v)
import json; json.dump({k:(v if not isinstance(v,dict) else {str(a):c for a,c in v.items()}) for k,v in PC.items()},open('/work/v8D_params.json','w'))
print('saved v8D')
```