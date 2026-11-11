Nearly all the remaining cycles have the same shape: an E=0 connection to a lone two-liberty stone next to a big opponent chain, often in games I was winning by 15–37 points. About half happen at the four corners. At a corner the whole pattern (lone stone, its liberties, and whether the adjacent opponent stone is a lone atari stone) lies within distance 2. I can therefore detect it exactly with 8 spare L6 units, one per corner-neighbour pair, with an exemption for genuine captures of a lone stone. I'll add these to the v8 structure with a prior midway between B and D, since those two tied.

```python
src=open('/work/build8.py').read()
src=src.replace("""    b[G1]=-Q['s0']; b[G2]=-Q['s1']""","""    b[G1]=-Q['s0']; b[G2]=-Q['s1']
    # corner cycle-start detectors (units 488..495)
    CORN=[]
    u=488
    for e in (0,8,72,80):
        for n in NB[e]:
            m=[x for x in NB[e] if x!=n][0]
            W[OWN+n,u]+=2; W[OPP+m,u]+=1
            for q in NB[n]:
                if q!=e: W[OPP+q,u]+=1; W[OWN+q,u]-=3
            W[NX0+m,u]-=2; W[NX1+m,u]-=2
            for r in NB[m]:
                if r!=e: W[OPP+r,u]+=2
            b[u]=-3
            CORN.append((u,e)); u+=1
    Q['_corn']=CORN""")
src=src.replace("""    b[81]=Q['pass_logit']
    T['out.W']=W""","""    for (u,e) in Q['_corn']: W[u,e]-=Q['w_corner']
    b[81]=Q['pass_logit']
    T['out.W']=W""")
src=src.replace("pass_logit=-5.0,","pass_logit=-5.0,w_corner=40.0,")
open('/work/build8e.py','w').write(src)
exec(src)
PE=dict(t1=2.5,W_o=0.3,prior={0:-2.0,1:1.3,2:1.5,3:0.7,4:0.9},prior_late={0:-1.0,1:0.4,2:0.55,3:0.35,4:0.35})
T=build(PE)
for k,v in T.items(): save(k,v)
import json; json.dump({k:(v if not isinstance(v,dict) else {str(a):c for a,c in v.items()}) for k,v in PE.items()},open('/work/v8E_params.json','w'))
print('saved v8E', T['out.W'][488:496].min(axis=1))
```