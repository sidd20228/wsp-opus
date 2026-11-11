Repetitions dropped to zero, but strength fell (56% / 69%, far fewer captures, more passing). The cause is a bug in the v16 eye-penalty change: the "+0.5 × weak neighbours" term also fires at ordinary points with two empty neighbours next to weak chains. It applied a −12 to −25 penalty to most moves next to my own uncertified chains, which are most chains. I'll re-derive the eye unit so that:
- an ordinary point with two empty neighbours never fires;
- a full eye and a near-eye are strongly penalised;
- an eye next to an atari chain is allowed only if the other neighbours are certified.

```python
src=open('/work/build16.py').read()
old=src[src.index("        # NE\n"):src.index("        # KO = relu")]
new='''        # NE = relu(na*(s_o - 2 s_x - deg) + nb - nc*a_o + nd*w)
        u=NE+e; b[u]=-Q['ne_a']*deg+Q['ne_b']
        for n in NB[e]:
            W[OWN+n,u]+=Q['ne_a']; W[OPP+n,u]-=2*Q['ne_a']; W[AO+n,u]-=Q['ne_c']
            W[OWN+n,u]+=Q['ne_d']; W[AO+n,u]-=Q['ne_d']; add_S3(u,n,-Q['ne_d'])
'''
src=src.replace(old,new)
src=src.replace("w_ne=25.0,ne_ao=2.0,ne_w=0.5,","w_ne=15.0,ne_a=3.0,ne_b=4.0,ne_c=4.0,ne_d=1.0,")
open('/work/build18.py','w').write(src)
exec(src)
# truth table check of NE
def NEv(s_o,s_x,deg,a_o,w):
    Q=DEFAULT; return max(0,Q['ne_a']*(s_o-2*s_x-deg)+Q['ne_b']-Q['ne_c']*a_o+Q['ne_d']*w)
for name,a in [('E2 w2',(2,0,4,0,2)),('E2 ao1 w1',(2,0,4,1,1)),('eye cert',(4,0,4,0,0)),('eye weak',(4,0,4,0,4)),('near cert',(3,0,4,0,0)),
               ('eye+atari+cert3',(4,0,4,1,0)),('eye+atari+weak3',(4,0,4,1,3)),('E1 sx1',(2,1,4,0,2)),('edge eye',(3,0,3,0,3)),('edge E2',(1,0,3,0,1)),('corner E1',(1,0,2,0,1))]:
    print(name,NEv(*a))
P18=dict(W_E=0.25,W_o=1.0,w_z=40.0)
T=build(P18)
for k,v in T.items(): save(k,v)
import json; json.dump(P18,open('/work/v18_params.json','w'))
print('saved v18')
```