Found an important flaw in position C: the capture that saves my stone in atari is blocked, because my "self-atari connect" detector also fires for an own stone in atari (it counts e(t)≤2). The repetition pattern it targets always involves a stone with exactly 2 liberties, while capture-to-save involves a stone with 1 liberty. I'll redefine Y to fire only for own isolated stones with ≥2 liberties (stones with ≥3 are cancelled by the ≥3-liberty witness anyway), then verify on both cases.

```python
src=open('/work/build8.py').read()
old="""        u=A.u('Y',p)
        iso=cfg.get('y_iso',1.0)
        for q in n: A.W[OWN(q),u]+=1-K*iso; A.W[OPP(q),u]+=1
        A.W[OWN(p),u]+=K; A.b[u]=3-deg-K"""
new="""        u=A.u('Y',p)
        iso=cfg.get('y_iso',1.0)
        if cfg.get('y_mode',0)==1:
            # Y = relu(e - 1) for own isolated stones (fires for e>=2, not for stones in atari)
            for q in n: A.W[OWN(q),u]+=-1-K*iso; A.W[OPP(q),u]+=-1
            A.W[OWN(p),u]+=K; A.b[u]=deg-1-K
        else:
            for q in n: A.W[OWN(q),u]+=1-K*iso; A.W[OPP(q),u]+=1
            A.W[OWN(p),u]+=K; A.b[u]=3-deg-K"""
assert old in src; src=src.replace(old,new); open('/work/build8.py','w').write(src)
exec(open('/work/build8.py').read())
import json
cfg=json.load(open('/work/cfg_v35.json')); cfg.update(y_mode=1)
arrs,layout=build8(cfg,inp)
for k,v in arrs.items(): save(k,v)
json.dump(cfg,open('/work/cfg_v36.json','w'))
G=json.load(open('/work/pass_41/games.json')); g=G[70]; n=len(g['moves'])
mv=[m['move'] for m in g['moves'][:n-3]]
d2=json.load(open('/work/tests2.json'))
P=[{"board":[".........",".........","...XO....","..XOXO...","..XO.O...","...XO....",".........",".........","........."],"to_move":"black"},
   {"moves":[("pass" if x==81 else x) for x in mv]}]
for gi,k,p,m2 in d2['SA'][:6]: P.append({"moves":[("pass" if x==81 else x) for x in m2]})
print('game70 next net move should NOT be 54; SA test moves:',[divmod(p,9) for gi,k,p,m2 in d2['SA'][:6]])
print(json.dumps({"positions":P,"activations":False},separators=(',',':')))
```