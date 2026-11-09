Endgames are fully played out; Black simply ends a few points short in the middle game. One principled improvement: the influence map counts weak stones (≤2 liberties, likely to die) at full strength. I already have per-stone ≤2-liberty flags at L5, so at L6 I can discount weak stones in the influence for free (linear terms).

```python
src=open('/work/build5.py').read()
old="""        for q in range(81):
            k=kern.get(mdist(p,q),0)
            if k:
                for uu in (ua,ub,uc): F.W[E.u('Z',q),uu]+=k
                s+=k"""
new="""        om=cfg.get('omega',0.0); oa=cfg.get('omega_at',0.0)
        for q in range(81):
            k=kern.get(mdist(p,q),0)
            if k:
                for uu in (ua,ub,uc):
                    F.W[E.u('Z',q),uu]+=k
                    F.W[E.u('L2M',q),uu]-=k*om
                    F.W[E.u('L2O',q),uu]+=k*om
                    F.W[E.u('AtO',q),uu]+=k*oa
                s+=k"""
assert old in src; src=src.replace(old,new); open('/work/build5.py','w').write(src)
exec(open('/work/build5.py').read())
import json
cfg=json.load(open('/work/cfg_v11.json')); cfg.update(omega=1.0, omega_at=1.0)
arrs,layout=build5(cfg,inp)
for k,v in arrs.items(): save(k,v)
json.dump(cfg,open('/work/cfg_v12.json','w')); json.dump(layout,open('/work/layout_v12.json','w'))
print(cfg)
```