Strong signal: Black's first move at a 4-4 point ((3,3),(3,5),(5,3),(5,5)) won 51/52, versus tengen 50/56. I'll add an "empty board" detector (one unit per layer — spare units exist) that makes Black's first move a 4-4 point.

```python
src=open('/work/build6.py').read()
src=src.replace("def build6(","def build7(")
# L1 add EMPTY unit
old="""    for nm in ['Z','G','EM','EO','Y']: A.add(nm)"""
new="""    for nm in ['Z','G','EM','EO','Y']: A.add(nm)
    A.add('EMP',1)
    uE=A.u('EMP',0)
    for q in range(81): A.W[OWN(q),uE]-=1; A.W[OPP(q),uE]-=1
    A.b[uE]=1"""
assert old in src; src=src.replace(old,new)
# carry in each subsequent layer: insert after each layer's add loop
for (lay,prev,anchor) in [('B','A',"for nm in ['Z','HO','VO','W3O','W3M','YB']: B.add(nm)"),
                          ('C','B',"for nm in ['Z','FO','F3O','F3M','YB']: C.add(nm)"),
                          ('D','C',"for nm in ['Z','FO','F3O','F3M','EYE','YB']: D.add(nm)"),
                          ('E','D',"for nm in ['Z','FO','L2O','L2M','SA','EY']: E.add(nm)"),
                          ('F','E',"for nm in ['TAC','PEN','BA','BB','BC','AtO']: F.add(nm)")]:
    assert anchor in src, anchor
    src=src.replace(anchor, anchor+f"\n    {lay}.add('EMP',1); {lay}.W[{prev}.u('EMP',0),{lay}.u('EMP',0)]=1")
old="""    ob[81]=cfg['P0']"""
new="""    for p in cfg.get('open_pts',[]): O[F.u('EMP',0),p]+=cfg.get('open_bonus',0.0)
    ob[81]=cfg['P0']"""
assert old in src; src=src.replace(old,new)
open('/work/build7.py','w').write(src)
exec(open('/work/build7.py').read())
import json
cfg=json.load(open('/work/cfg_v14.json')); cfg.update(open_pts=[30,32,48,50], open_bonus=20.0)
arrs,layout=build7(cfg,inp)
for k,v in arrs.items(): save(k,v)
json.dump(cfg,open('/work/cfg_v16.json','w')); json.dump(layout,open('/work/layout_v16.json','w'))
print([ (i,L.get('EMP')) for i,L in enumerate(layout)])
print(cfg)
```