I'll compare each test position's choice with the intended behaviour.

```python
import json, numpy as np
t=json.load(open('/work/test16.json')); P=np.load('/work/pass_16/probs.npy'); L=np.load('/work/pass_16/logits.npy')
for i,(pk,gi,ply,note) in enumerate(t['meta']):
    top=np.argsort(-P[i])[:3]
    print(i,pk,gi,ply,note,'| top',[(divmod(int(x),9) if x<81 else 'pass',round(float(P[i,x]),3)) for x in top])
```