Two concrete tactical errors from the late game:
- **Game 15, snapback.** The net captured a single stone to save its chain in atari, from a point with no empty neighbour. The chain was left with one liberty, and the opponent recaptured the whole chain. This is the ko shape with my atari stones counted as surrounding stones, so I'll fold own-atari neighbours into the KO unit.
- **Game 3, missed connection.** The net didn't fill an eye point to save a stone in atari, though that was safe because the group had certified liberties. I'll re-enable the eye relaxation for atari neighbours, but add a penalty when the surrounding own chains are only weak (uncertified). That keeps the 32-stone disaster blocked.

```python
src=open('/work/build14.py').read()
src=src.replace("""        u=KO+e; b[u]=2-deg
        for n in NB[e]: W[OPP+n,u]+=1; W[NXP+n,u]-=1""","""        u=KO+e; b[u]=2-deg
        for n in NB[e]: W[OPP+n,u]+=1; W[NXP+n,u]-=1; W[AO+n,u]+=Q['ko_ao']""")
src=src.replace("""        for n in NB[e]: W[OWN+n,u]+=1; W[OPP+n,u]-=2; W[AO+n,u]-=Q['ne_ao']""","""        for n in NB[e]:
            W[OWN+n,u]+=1; W[OPP+n,u]-=2; W[AO+n,u]-=Q['ne_ao']
            # + ne_w * w,  w = OWN - AO - S3
            W[OWN+n,u]+=Q['ne_w']; W[AO+n,u]-=Q['ne_w']; add_S3(u,n,-Q['ne_w'])""")
src=src.replace("w_ne=25.0,ne_ao=1.0,","w_ne=25.0,ne_ao=2.0,ne_w=0.5,ko_ao=1.0,")
assert "ko_ao" in src and "ne_w" in src
open('/work/build16.py','w').write(src)
exec(src)
P16=dict(W_E=0.25,W_o=1.0)
T=build(P16)
for k,v in T.items(): save(k,v)
import json; json.dump(P16,open('/work/v16_params.json','w'))
print('saved v16')
```