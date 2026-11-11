A leftover reference to the old self-atari parameters remains. I'll fix it and rebuild.

```python
src=open('/work/build11.py').read()
src=src.replace("iw=Q['infw']; sa=Q['sa']","iw=Q['infw']")
open('/work/build11.py','w').write(src)
exec(src)
P11=dict(t1=2.5,W_o=0.3,prior={0:-2.0,1:1.0,2:1.6,3:0.8,4:1.0},prior_late={0:-1.0,1:0.3,2:0.6,3:0.4,4:0.4},gamma=0.3,W_cap=12.0)
T=build(P11)
for k,v in T.items(): save(k,v)
i=src.index("    iw=Q['infw']")
print(src[i:i+2200])
L=DEFAULT['lone']; Cn=DEFAULT['conn']
def lone(E,ax,so): return max(0,L['k0']-L['E']*E-L['ax']*ax-L['mu']*so)
def conn(E,ao,w,t,ax): return max(0,Cn['ao']*ao+Cn['w']*w-Cn['E']*E-Cn['t']*t-Cn['ax']*ax)
print('lone: E1',lone(1,0,0),'ko',lone(0,1,0),'eyekill2',lone(0,2,0),'E1cap',lone(1,1,0),'E2',lone(2,0,0),'E1 so1',lone(1,0,1))
print('conn: E1 ao1',conn(1,1,0,0,0),'E1 w1',conn(1,0,1,0,0),'E0 w1',conn(0,0,1,0,0),'E0 w1 ax1',conn(0,0,1,0,1),'E0 w2',conn(0,0,2,0,0),
      'E2 ao1',conn(2,1,0,0,0),'E0 ao1 t1',conn(0,1,0,1,0),'capsave E0 ao1 ax1',conn(0,1,0,0,1),'E1 t1',conn(1,0,0,1,0),'E0 t1',conn(0,0,0,1,0))
import json
def ser(P): return {k:(v if not isinstance(v,dict) else {str(a):c for a,c in v.items()}) for k,v in P.items()}
json.dump(ser(P11),open('/work/v11_params.json','w'))
```