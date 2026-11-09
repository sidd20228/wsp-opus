A recurring pattern: the right move is often top-ranked but with only 25–45% probability, and a worse move gets sampled (e.g. move 68 here, (1,8) would have saved the group). I'll sharpen the whole policy by scaling all logits ×2 (equivalent to temperature 0.5; the pass decision boundary is unchanged).

```python
src=open('/work/build5.py').read()
if "lam=cfg.get('lam',1.0)" not in src:
    src=src.replace("    ob[81]=cfg['P0']\n    layers=[A,B,C,D,E,F]; out={}",
                    "    ob[81]=cfg['P0']\n    lam=cfg.get('lam',1.0); O*=lam; ob*=lam\n    layers=[A,B,C,D,E,F]; out={}")
    open('/work/build5.py','w').write(src)
exec(open('/work/build5.py').read())
import json
cfg=json.load(open('/work/cfg_v10.json')); cfg['lam']=2.0
arrs,layout=build5(cfg,inp)
for k,v in arrs.items(): save(k,v)
json.dump(cfg,open('/work/cfg_v11.json','w')); json.dump(layout,open('/work/layout_v11.json','w'))
print(cfg, float(np.abs(arrs['out.W']).max()))
```