The cap on the capture exemption fails because it counts adjacent stones, not chains. A big chain touching e with three stones exempts a fake-capture connection completely. The robust distinction: a move touching none of my stones can only lose that one stone, and a legal no-liberty lone move must truly capture. Only connecting moves create cycles. So I split self-atari into two units:
- **SA_lone** (no own neighbours): penalises a single-stone self-atari. A ko-type capture (one adjacent atari stone, no empty neighbour) gets a heavy penalty, so the separate KO unit is no longer needed. Multi-stone eye-filling kills get no penalty.
- **SA_conn** (connecting moves): penalises futile atari extensions and E=0/E=1 connections to weak chains, with almost no capture exemption.

```python
src=open('/work/build8.py').read()
src=src.replace("LIN,SA,NE,INFP,N5,KO=0,81,162,243,324,405","LIN,SA,NE,INFP,N5,SAC=0,81,162,243,324,405")
old_sa=src[src.index("        u=SA+e; k0,cE"):src.index("        u=NE+e")]
new_sa='''        # SA_lone = relu(k0 - cE*E - cax*a_x - mu*s_o)   (E = deg - s_o - s_x)
        L=Q['lone']; u=SA+e; b[u]=L['k0']-L['E']*deg
        for n in NB[e]:
            W[OWN+n,u]+=L['E']-L['mu']; W[OPP+n,u]+=L['E']
            W[NX0+n,u]-=L['ax']; W[NX1+n,u]-=L['ax']
        # SA_conn = relu(ca*a_o + cw*w - cE*E - ct*t - cx*a_x),  w = s_o - a_o - t
        Cn=Q['conn']; u=SAC+e; b[u]=-Cn['E']*deg
        for n in NB[e]:
            W[OWN+n,u]+=Cn['w']+Cn['E']; W[OPP+n,u]+=Cn['E']
            W[AO+n,u]+=Cn['ao']-Cn['w']; W[S3+n,u]+=-Cn['t']-Cn['w']
            W[NX0+n,u]-=Cn['ax']; W[NX1+n,u]-=Cn['ax']
'''
src=src.replace(old_sa,new_sa)
old_ko=src[src.index("        u=KO+e; b[u]=2-deg"):src.index("        b[INFP+e]=-Q['t1']")]
src=src.replace(old_ko,"")
src=src.replace("        W[KO+e,e]=-Q['w_ko']\n","        W[SAC+e,e]=-Q['w_conn']\n")
src=src.replace("sa=dict(k0=4.0,E=2.0,t=4.0,w=1.0,ax=2.0))",
 "lone=dict(k0=10.0,E=8.0,ax=5.0,mu=10.0),conn=dict(ao=4.0,w=3.0,E=2.0,t=4.0,ax=1.0),w_conn=6.0)")
open('/work/build11.py','w').write(src)
exec(src)
P11=dict(t1=2.5,W_o=0.3,prior={0:-2.0,1:1.0,2:1.6,3:0.8,4:1.0},prior_late={0:-1.0,1:0.3,2:0.6,3:0.4,4:0.4},gamma=0.3,W_cap=12.0)
T=build(P11)
for k,v in T.items(): save(k,v)
i=src.index('    # ---------- L6') if '    # ---------- L6' in src else src.index("    W=np.zeros((512,512),np.float32); b=np.zeros(512,np.float32)\n    iw=Q['infw']")
print(src[i:i+2600])
# sanity tables
L=DEFAULT['lone']; Cn=DEFAULT['conn']
def lone(E,ax,so): return max(0,L['k0']-L['E']*E-L['ax']*ax-L['mu']*so)
def conn(E,ao,w,t,ax): return max(0,Cn['ao']*ao+Cn['w']*w-Cn['E']*E-Cn['t']*t-Cn['ax']*ax)
print('lone: E1',lone(1,0,0),'ko',lone(0,1,0),'eyekill2',lone(0,2,0),'E1cap',lone(1,1,0),'E2',lone(2,0,0),'E1 so1',lone(1,0,1))
print('conn: E1 ao1',conn(1,1,0,0,0),'E1 w1',conn(1,0,1,0,0),'E0 w1',conn(0,0,1,0,0),'E0 w1 ax1',conn(0,0,1,0,1),'E0 w2',conn(0,0,2,0,0),
      'E2 ao1',conn(2,1,0,0,0),'E0 ao1 t1',conn(0,1,0,1,0),'capsave E0 ao1 ax1',conn(0,1,0,0,1),'E1 t1',conn(1,0,0,1,0),'E0 t1',conn(0,0,0,1,0))
```