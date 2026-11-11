The mechanics work: capture, escape and the long-chain capture are all correct. The P7 false atari is the known limit of the 4-hop design. Positional logits are too flat, though: the first move only reaches 5% on tengen. I'll sharpen them a bit, then get a baseline from games.

```python
exec(open('/work/build.py').read())
P=dict(prior={0:-3.0,1:-1.0,2:1.0,3:1.4,4:2.4},W_E=0.5,W_c=1.5)
T=build(P)
for k,v in T.items(): save(k,v)
json_P=P
import json; json.dump({k:(v if not isinstance(v,dict) else {str(a):b for a,b in v.items()}) for k,v in P.items()},open('/work/v2_params.json','w'))
print('saved v2')
```