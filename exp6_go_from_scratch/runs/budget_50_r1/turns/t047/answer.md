The net plays odd first-line moves mid-game (e.g. (0,0), (4,0), (8,0)) because my convolved "decidedness penalty" treats off-board points as undecided, which favours edges. I'll replace it with a first-order territory-swing estimate: the output sums a "contestedness" bump relu(θ−|I|) over nearby points (built from three ReLUs of the influence), plus a penalty for playing inside own territory. Off-board points then contribute nothing, and the empty board naturally prefers the center.

```python
src=open('/work/build3.py').read()
src=src.replace("def build3(cfg, inp):","def build4(cfg, inp):")
src=src.replace("for nm in ['AtO','AtM','PEN','IP','IN']: F.add(nm)","for nm in ['AtO','AtM','PEN','BA','BB','BC']: F.add(nm)")
src=src.replace("""        up=F.u('IP',p); un=F.u('IN',p); s=0
        for q in range(81):
            k=kern.get(mdist(p,q),0)
            if k: F.W[E.u('Z',q),up]+=k; F.W[E.u('Z',q),un]-=k; s+=k
        F.b[up]=-2*s-th; F.b[un]=2*s-th""",
"""        ua=F.u('BA',p); ub=F.u('BB',p); uc=F.u('BC',p); s=0
        for q in range(81):
            k=kern.get(mdist(p,q),0)
            if k:
                for uu in (ua,ub,uc): F.W[E.u('Z',q),uu]+=k
                s+=k
        F.b[ua]=-2*s+th; F.b[ub]=-2*s; F.b[uc]=-2*s-th""")
src=src.replace("""        for q in range(81):
            dd=mdist(p,q)
            if dd in ck:
                O[F.u('IP',q),p]-=sp*ck[dd]*cfg['own_scale']
                O[F.u('IN',q),p]-=sp*ck[dd]*cfg['opp_scale']""",
"""        for q in range(81):
            dd=mdist(p,q)
            if dd in ck:
                w=sp*cfg['alpha']*ck[dd]
                O[F.u('BA',q),p]+=w; O[F.u('BB',q),p]-=2*w; O[F.u('BC',q),p]+=w
        O[F.u('BC',p),p]-=sp*cfg['beta_own']""")
open('/work/build4.py','w').write(src)
exec(open('/work/build4.py').read())
import json
cfg=json.load(open('/work/cfg_v6.json'))
cfg.update(theta=1.5, conv={0:1.0,1:1.0,2:0.5,3:0.25}, alpha=0.5, beta_own=2.0, spos=2.0)
arrs,layout=build4(cfg,inp)
for k,v in arrs.items(): save(k,v)
json.dump(cfg,open('/work/cfg_v7.json','w')); json.dump(layout,open('/work/layout_v7.json','w'))
print(layout[5]); print(cfg)
```