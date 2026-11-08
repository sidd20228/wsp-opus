Let me verify v4's new self-atari logic on the same 50 test positions before spending a games pass.

```python
import json
d=json.load(open('/work/tests2.json'))
posl=[{"board":b,"to_move":tm} for (_,b,tm) in d['T']]+[{"moves":[("pass" if x==81 else x) for x in m]} for (_,m) in d['R']]
for gi,k,p,mv in d['SA']: posl.append({"moves":[("pass" if x==81 else x) for x in mv]})
s=json.dumps({"positions":posl,"activations":True},separators=(',',':'))
open('/work/pos_req_tests2.json','w').write(s); print(len(s))
```