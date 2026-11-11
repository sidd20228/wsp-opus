The result is identical to pass 9, down to the margin and stone counts, so practice games are deterministic for a given network. Re-running a network gives no new information, and every comparison between versions is one sample from a different trajectory. v8B stays the best observed.

I'll use the remaining passes on careful variants of v8B. The first, v8C, adds the two clearly motivated tactical fixes to v8B's structure:
- the corrected eye/near-eye unit, which blocks eye-fills next to atari chains when the rest of the group is uncertified (the 32-stone disaster);
- the KO unit counting own atari neighbours, which avoids snapback-prone capture-saves.

```python
src=open('/work/build8.py').read()
old=src[src.index("        u=NE+e; b[u]=2-deg"):src.index("        u=KO+e")]
new='''        u=NE+e; b[u]=-Q['ne_a']*deg+Q['ne_b']
        for n in NB[e]:
            W[OWN+n,u]+=Q['ne_a']; W[OPP+n,u]-=2*Q['ne_a']; W[AO+n,u]-=Q['ne_c']
            # + ne_d * w,  w = OWN - AO - S3
            W[OWN+n,u]+=Q['ne_d']; W[AO+n,u]-=Q['ne_d']; W[S3+n,u]-=Q['ne_d']
'''
src=src.replace(old,new)
src=src.replace("        for n in NB[e]: W[OPP+n,u]+=1; W[NX0+n,u]-=1; W[NX1+n,u]-=1",
                "        for n in NB[e]: W[OPP+n,u]+=1; W[NX0+n,u]-=1; W[NX1+n,u]-=1; W[AO+n,u]+=Q['ko_ao']")
src=src.replace("w_ne=25.0,","w_ne=15.0,ne_a=3.0,ne_b=4.0,ne_c=4.0,ne_d=1.0,ko_ao=1.0,")
assert "ko_ao']" in src and "ne_a" in src
open('/work/build8c.py','w').write(src)
exec(src)
PB=dict(t1=2.5,W_o=0.3,prior={0:-2.0,1:1.0,2:1.6,3:0.8,4:1.0},prior_late={0:-1.0,1:0.3,2:0.6,3:0.4,4:0.4})
T=build(PB)
for k,v in T.items(): save(k,v)
i=src.index("        u=NE+e"); print(src[i:i+900])
```