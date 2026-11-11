The strong opponent plays an almost fixed opening: (5,3), (5,1), (6,1), (6,2), (7,2), (4,7), (1,5)… That is low, territorial, solidly connected play. My positional component is weak, and its first moves are near-random. Fixes plus an A/B test:
- **Eye unit:** switch off near-eye/eye penalties when an adjacent own chain is in atari.
- **KO unit:** require exactly one adjacent atari stone, so whole-group kills by eye-filling aren't blocked.
- **Solid shape:** raise the deep-territory threshold and add a small bonus for adjacent own stones.
- **Prior A/B:** variant A keeps the center-heavy prior, variant B favours the 3rd and 2nd lines.

```python
src=open('/work/build7.py').read()
src=src.replace("""        u=NE+e; b[u]=2-deg
        for n in NB[e]: W[OWN+n,u]+=1; W[OPP+n,u]-=2""","""        u=NE+e; b[u]=2-deg
        for n in NB[e]: W[OWN+n,u]+=1; W[OPP+n,u]-=2; W[AO+n,u]-=2""")
src=src.replace("""        u=KO+e; b[u]=1-deg
        for n in NB[e]: W[OPP+n,u]+=1""","""        u=KO+e; b[u]=2-deg
        for n in NB[e]: W[OPP+n,u]+=1; W[NX0+n,u]-=1; W[NX1+n,u]-=1""")
assert "W[AO+n,u]-=2" in src and "W[NX0+n,u]-=1; W[NX1+n,u]-=1" in src
open('/work/build8.py','w').write(src)
exec(src)
PA=dict(t1=2.5,W_o=0.3)
PB=dict(t1=2.5,W_o=0.3,prior={0:-2.0,1:1.0,2:1.6,3:0.8,4:1.0},prior_late={0:-1.0,1:0.3,2:0.6,3:0.4,4:0.4})
import json
json.dump({'A':PA,'B':{k:(v if not isinstance(v,dict) else {str(a):c for a,c in v.items()}) for k,v in PB.items()}},open('/work/v8_params.json','w'))
T=build(PA)
for k,v in T.items(): save(k,v)
print('saved v8A')
```