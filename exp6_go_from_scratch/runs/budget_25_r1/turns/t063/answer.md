The pattern: when my net passes, the opponent attacks its weak boundary stones and stones inside the opponent's area, and the net never responds. In game 49, about 15 points of area were lost to a series of moves met with passes. For v12 I'll go back to the v8B structure and make two changes:
- Lower the pass logit to −9, so harmless own-territory or defensive moves beat passing. Kos are no longer taken, so passing less won't trigger the double-ko issue.
- Add a bonus for moves that extend my weak, uncertified chains.

```python
exec(open('/work/build8.py').read())
src=open('/work/build8.py').read()
src=src.replace("            W[AO+n,u]+=Q['W_esc']\n","            W[AO+n,u]+=Q['W_esc']\n            W[OWN+n,u]+=Q['W_def']; W[AO+n,u]-=Q['W_def']; W[S3+n,u]-=Q['W_def']\n",1)
src=src.replace("pass_logit=-5.0,","pass_logit=-5.0,W_def=0.0,")
open('/work/build12.py','w').write(src)
exec(src)
P12=dict(t1=2.5,W_o=0.3,prior={0:-2.0,1:1.0,2:1.6,3:0.8,4:1.0},prior_late={0:-1.0,1:0.3,2:0.6,3:0.4,4:0.4},pass_logit=-9.0,W_def=1.0)
T=build(P12)
for k,v in T.items(): save(k,v)
import json
def ser(P): return {k:(v if not isinstance(v,dict) else {str(a):c for a,c in v.items()}) for k,v in P.items()}
json.dump(ser(P12),open('/work/v12_params.json','w'))
print('saved v12; LIN lower bound ok since W_def*w>=0 (w>=0 for empty e)')
```