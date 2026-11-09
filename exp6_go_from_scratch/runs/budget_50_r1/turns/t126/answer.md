Found a real bug: in game 103 my net filled its own (false) eye at (5,3), killing a 16-stone group, because the "defend a ≤2-liberty group" bonus (8 per adjacent stone ×4 neighbours) outweighed the self-atari penalty — even though filling a point with no empty neighbours adds no liberties. Fix: gate the whole tactics unit (atari-moves and defence bonuses) by the self-atari indicator, so these bonuses vanish at self-atari points (real captures are a separate unit and unaffected).

```python
src=open('/work/build8.py').read()
old="""        u=F.u('TAC',p)
        for q in nb(p):
            F.W[E.u('L2O',q),u]+=cfg['w_at']
            F.W[E.u('L2M',q),u]+=cfg['w_esc']"""
new="""        u=F.u('TAC',p)
        for q in nb(p):
            F.W[E.u('L2O',q),u]+=cfg['w_at']
            F.W[E.u('L2M',q),u]+=cfg['w_esc']
        F.W[E.u('SA',p),u]-=cfg.get('tac_sa_gate',0.0)"""
assert old in src; src=src.replace(old,new); open('/work/build8.py','w').write(src)
exec(open('/work/build8.py').read())
import json
cfg=json.load(open('/work/cfg_v26.json')); cfg.update(tac_sa_gate=100.0)
arrs,layout=build8(cfg,inp)
for k,v in arrs.items(): save(k,v)
json.dump(cfg,open('/work/cfg_v27.json','w'))
print(cfg)
```