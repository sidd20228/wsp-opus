Black's "peaceful" games are close to a coin flip, so territorial play is what matters. One suspect: the static line bias (first line −12 logits) is meant for the opening but also suppresses first-line endgame moves (hane/descents), which are often the biggest. I'll make the line bias fade with the number of stones on the board (one global unit carried through the layers).

```python
src=open('/work/build6.py').read().replace("def build6(","def build8(")
old="""    for nm in ['Z','G','EM','EO','Y']: A.add(nm)"""
new="""    for nm in ['Z','G','EM','EO','Y']: A.add(nm)
    A.add('PH',1); uP=A.u('PH',0); S0=cfg.get('S0',40.0)
    for q in range(81): A.W[OWN(q),uP]-=1.0/S0; A.W[OPP(q),uP]-=1.0/S0
    A.b[uP]=1.0"""
assert old in src; src=src.replace(old,new)
for (lay,prev,anchor) in [('B','A',"for nm in ['Z','HO','VO','W3O','W3M','YB']: B.add(nm)"),
                          ('C','B',"for nm in ['Z','FO','F3O','F3M','YB']: C.add(nm)"),
                          ('D','C',"for nm in ['Z','FO','F3O','F3M','EYE','YB']: D.add(nm)"),
                          ('E','D',"for nm in ['Z','FO','L2O','L2M','SA','EY']: E.add(nm)"),
                          ('F','E',"for nm in ['TAC','PEN','BA','BB','BC','AtO']: F.add(nm)")]:
    assert anchor in src, anchor
    src=src.replace(anchor, anchor+f"\n    {lay}.add('PH',1); {lay}.W[{prev}.u('PH',0),{lay}.u('PH',0)]=1")
old="""        ob[p]=sp*cfg['bias'][line(p)]"""
new="""        bs=cfg.get('bias_static',cfg['bias'])
        ob[p]=sp*bs[line(p)]
        O[F.u('PH',0),p]+=sp*(cfg['bias'][line(p)]-bs[line(p)])"""
assert old in src; src=src.replace(old,new)
open('/work/build8.py','w').write(src)
exec(open('/work/build8.py').read())
import json
cfg=json.load(open('/work/cfg_v14.json')); cfg.update(S0=40.0, bias_static=[-0.5,0.0,0.0,0.0,0.0])
arrs,layout=build8(cfg,inp)
for k,v in arrs.items(): save(k,v)
json.dump(cfg,open('/work/cfg_v19.json','w')); json.dump(layout,open('/work/layout_v19.json','w'))
print([L.get('PH') for L in layout]); print(cfg)
```