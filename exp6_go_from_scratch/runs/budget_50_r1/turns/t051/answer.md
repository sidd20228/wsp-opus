In game 33 the only non-terrible move was (2,0): connecting the ko, which is a *false* eye — but my eye detector still gives it 0.5 (penalty 100 > pass). I'll sharpen the eye formula so false/uncertain eyes score 0 (edge: any opponent diagonal; center: own−opp diagonal balance ≤ 0) while true eyes keep a huge penalty. Then I'll A/B the two positional schemes with 128 games each against "strong" for a lower-noise comparison.

```python
for fn in ['/work/build3.py','/work/build4.py']:
    src=open(fn).read()
    old="""        beta=0.25 if deg==4 else 0.5
        for q in n: D.W[C.u('Z',q),u]+=2
        for q in g: D.W[C.u('Z',q),u]+=beta
        D.b[u]=-6*deg+0.5-2*beta*len(g)"""
    new="""        a_,b_={4:(0.3,0.0),3:(0.75,-0.5),2:(0.5,0.5)}[deg]
        for q in n: D.W[C.u('Z',q),u]+=2
        for q in g: D.W[C.u('Z',q),u]+=a_
        D.b[u]=-6*deg+b_-2*a_*len(g)"""
    assert old in src
    open(fn,'w').write(src.replace(old,new))
exec(open('/work/build3.py').read())
import json
cfg=json.load(open('/work/cfg_v6.json')); cfg['w_eye']=300.0
arrs,layout=build3(cfg,inp)
for k,v in arrs.items(): save(k,v)
json.dump(cfg,open('/work/cfg_v8.json','w')); json.dump(layout,open('/work/layout_v8.json','w'))
print('saved v8 (v6 positional + new eyes)', cfg)
```