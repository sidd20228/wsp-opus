Real captures outnumber false ones roughly 4:1 at those points, so the broad detector would block too many good captures in capturing races. I'll keep v11's narrower rule. A cheap improvement instead: move the opponent-atari indicator back to L6 (flood radius 4 instead of 3), summed over neighbors in the output layer — L6 has room.

```python
src=open('/work/build5.py').read()
# L5: replace AtO with FO flood step; L6: add AtO computed from FO3
src=src.replace("for nm in ['Z','AtO','L2O','L2M','SA','EY']: E.add(nm)","for nm in ['Z','FO','L2O','L2M','SA','EY']: E.add(nm)")
old="""        for nm,fl,sg in (('AtO','FO',-1),('L2O','F3O',-1),('L2M','F3M',1)):"""
new="""        for nm,fl,sg in (('L2O','F3O',-1),('L2M','F3M',1)):"""
assert old in src; src=src.replace(old,new)
old="""    kap=cfg['kap']
    for p in range(81):"""
new="""    kap=cfg['kap']
    flood(E,D,'FO',['FO'],-1,4096)
    for p in range(81):"""
assert old in src; src=src.replace(old,new)
src=src.replace("for nm in ['TAC','PEN','BA','BB','BC']: F.add(nm)","for nm in ['TAC','PEN','BA','BB','BC','AtO']: F.add(nm)")
old="""        for q in nb(p):
            F.W[E.u('AtO',q),u]+=cfg['w_cap']-cfg['w_at']
            F.W[E.u('L2O',q),u]+=cfg['w_at']"""
new="""        for q in nb(p):
            F.W[E.u('L2O',q),u]+=cfg['w_at']"""
assert old in src; src=src.replace(old,new)
old="""        u=F.u('PEN',p); F.W[E.u('SA',p),u]=cfg['w_sa']; F.W[E.u('EY',p),u]=1"""
new="""        u=F.u('PEN',p); F.W[E.u('SA',p),u]=cfg['w_sa']; F.W[E.u('EY',p),u]=1
        u=F.u('AtO',p)
        for q in [p]+nb(p): F.W[E.u('FO',q),u]-=1
        F.b[u]+=1; gate_unit(F,E,u,p,-1,4)"""
assert old in src; src=src.replace(old,new)
# omega terms referenced E.u('AtO') -> use F? remove omega_at usage (unused in v11)
src=src.replace("                    F.W[E.u('AtO',q),uu]+=k*oa\n","")
old="""        O[F.u('TAC',p),p]+=1"""
new="""        O[F.u('TAC',p),p]+=1
        for q in nb(p): O[F.u('AtO',q),p]+=cfg['w_cap']-cfg['w_at']"""
assert old in src; src=src.replace(old,new)
open('/work/build6.py','w').write(src.replace('def build5(','def build6('))
exec(open('/work/build6.py').read())
import json
cfg=json.load(open('/work/cfg_v11.json'))
arrs,layout=build6(cfg,inp)
for k,v in arrs.items(): save(k,v)
json.dump(cfg,open('/work/cfg_v14.json','w')); json.dump(layout,open('/work/layout_v14.json','w'))
print(layout[4],layout[5]); print(cfg)
```