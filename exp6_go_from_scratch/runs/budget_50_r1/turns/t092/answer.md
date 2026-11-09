Here the net spent ~5 moves invading White's lightly-enclosed bottom-left corner (stones that died) while boundary moves elsewhere waited. Cause: influence near edges/corners is computed from fewer on-board points, so the opponent's corners look "contested". I'll normalize influence by the on-board kernel mass (partial, exponent 0.5) so edge/corner territory is recognized properly.

```python
src=open('/work/build6.py').read()
old="""        ua=F.u('BA',p); ub=F.u('BB',p); uc=F.u('BC',p); s=0
        om=cfg.get('omega',0.0); oa=cfg.get('omega_at',0.0)
        for q in range(81):
            k=kern.get(mdist(p,q),0)
            if k:
                for uu in (ua,ub,uc):
                    F.W[E.u('Z',q),uu]+=k
                    F.W[E.u('L2M',q),uu]-=k*om
                    F.W[E.u('L2O',q),uu]+=k*om
                s+=k"""
new="""        ua=F.u('BA',p); ub=F.u('BB',p); uc=F.u('BC',p); s=0
        om=cfg.get('omega',0.0)
        Mp=sum(kern.get(mdist(p,q),0) for q in range(81)); Mc=sum(kern.get(mdist(40,q),0) for q in range(81))
        fn=(Mc/Mp)**cfg.get('gnorm',0.0)
        for q in range(81):
            k=kern.get(mdist(p,q),0)*fn
            if k:
                for uu in (ua,ub,uc):
                    F.W[E.u('Z',q),uu]+=k
                    F.W[E.u('L2M',q),uu]-=k*om
                    F.W[E.u('L2O',q),uu]+=k*om
                s+=k"""
assert old in src; src=src.replace(old,new); open('/work/build6.py','w').write(src)
exec(open('/work/build6.py').read())
import json
cfg=json.load(open('/work/cfg_v14.json')); cfg.update(gnorm=0.5)
arrs,layout=build6(cfg,inp)
for k,v in arrs.items(): save(k,v)
json.dump(cfg,open('/work/cfg_v18.json','w'))
print(cfg)
```