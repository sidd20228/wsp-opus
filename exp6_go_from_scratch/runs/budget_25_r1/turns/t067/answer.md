Game 2 shows the danger of my "disable eye penalty when a neighbour is in atari" fix. The net filled one of the two eyes of a 32-stone group to rescue a single atari stone, and the whole group was captured. The group's liberties weren't certified, so the self-atari unit gave only a mild penalty. For v13:
- **Eye penalty:** only half-relaxed for eyes next to an atari chain (fully relaxed for near-eyes). Full eyes keep −25.
- **Pass logit:** set to −7, between the two tested values.
- **Positional terms:** all scaled by 1.5, to test whether more decisive positional choices help.

```python
src=open('/work/build12.py').read()
src=src.replace("W[OWN+n,u]+=1; W[OPP+n,u]-=2; W[AO+n,u]-=2","W[OWN+n,u]+=1; W[OPP+n,u]-=2; W[AO+n,u]-=Q['ne_ao']")
src=src.replace("W_def=0.0,","W_def=0.0,ne_ao=2.0,")
open('/work/build13.py','w').write(src)
exec(src)
s=1.5
P13=dict(t1=2.5,W_o=0.3*s,prior={k:v*s for k,v in {0:-2.0,1:1.0,2:1.6,3:0.8,4:1.0}.items()},
         prior_late={k:v*s for k,v in {0:-1.0,1:0.3,2:0.6,3:0.4,4:0.4}.items()},
         pass_logit=-7.0,W_def=1.0,ne_ao=1.0,W_E=0.5*s,W_c=1.5*s,lin_inf=0.3*s,alpha=3.0*s)
T=build(P13)
for k,v in T.items(): save(k,v)
import json
def ser(P): return {k:(v if not isinstance(v,dict) else {str(a):c for a,c in v.items()}) for k,v in P.items()}
json.dump(ser(P13),open('/work/v13_params.json','w'))
print('saved v13')
```