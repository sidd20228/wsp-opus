The opening is nearly uniform among 3rd-line points: the top move has 3% probability. That's a lever for later. First I'll apply the new self-atari constants and reduce the connectivity discount on top of variant B, to remove the remaining repetitions without losing real captures.

```python
exec(open('/work/build8.py').read())
PB=dict(t1=2.5,W_o=0.3,prior={0:-2.0,1:1.0,2:1.6,3:0.8,4:1.0},prior_late={0:-1.0,1:0.3,2:0.6,3:0.4,4:0.4})
P9=dict(PB); P9.update(gamma=0.2,sa=dict(k0=7.0,E=5.0,t=7.0,w=1.0,ax=2.5))
T=build(P9)
for k,v in T.items(): save(k,v)
import json
def ser(P): return {k:(v if not isinstance(v,dict) else {str(a):c for a,c in v.items()}) for k,v in P.items()}
json.dump(ser(P9),open('/work/v9_params.json','w'))
# sanity table of SA values
sa=P9['sa']
def SA(E,t,w,ax): return max(0,sa['k0']-sa['E']*E-sa['t']*t-sa['w']*w-sa['ax']*ax)
for name,args in [('E1 single selfatari',(1,0,0,0)),('E1 extend atari(a_o=1)',(1,0,0,0)),('E1 w1',(1,0,1,0)),('E0 w1',(0,0,1,0)),
                  ('E0 w1 fakecap',(0,0,1,1)),('E1 capture',(1,0,0,1)),('ko',(0,0,0,1)),('E0 eyekill2',(0,0,0,2)),('E2',(2,0,0,0)),('E1 t1',(1,1,0,0)),('E0 t1',(0,1,0,0))]:
    print(name,SA(*args))
```