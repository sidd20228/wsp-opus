Game 94 lost a +22.5 position by repetition: a false-capture lure led to a 3-stone self-atari (connecting to a non-isolated 2-liberty group at a point with no empty neighbors), then capture/recapture repeated the position. I'll generalize the detector: any own neighbor stone with ≤2 own liberties (not just isolated ones) at an ε=0 point triggers the heavy self-atari penalty, still cancelled when a neighboring group has a ≥3-liberty witness.

```python
src=open('/work/build5.py').read()
old="""        u=A.u('Y',p)
        for q in n: A.W[OWN(q),u]+=1-K; A.W[OPP(q),u]+=1"""
new="""        u=A.u('Y',p)
        iso=cfg.get('y_iso',1.0)
        for q in n: A.W[OWN(q),u]+=1-K*iso; A.W[OPP(q),u]+=1"""
assert old in src; src=src.replace(old,new)
old2="""        B.W[A.u('G',p),u]=-8
    C=Lay(512)"""
new2="""        B.W[A.u('G',p),u]=-16
    C=Lay(512)"""
assert old2 in src; src=src.replace(old2,new2)
old3="for q in nb(p): E.W[D.u('F3M',q),u]-=(2+8*kap)"
assert old3 in src; src=src.replace(old3,"for q in nb(p): E.W[D.u('F3M',q),u]-=(2+16*kap)")
open('/work/build5.py','w').write(src)
exec(open('/work/build5.py').read())
import json
cfg=json.load(open('/work/cfg_v11.json')); cfg['y_iso']=0.0
arrs,layout=build5(cfg,inp)
for k,v in arrs.items(): save(k,v)
json.dump(cfg,open('/work/cfg_v13.json','w')); json.dump(layout,open('/work/layout_v13.json','w'))
print(cfg)
```