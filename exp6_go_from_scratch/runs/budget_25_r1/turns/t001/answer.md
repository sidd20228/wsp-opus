I'll start by writing a weight builder from my own design. The plan:

- **Layers L1–L5:** a chain-liberty analyser. It uses parity-coded liberty flags propagated along chains.
- **Layer L6:** per-move tactical and positional features.
- **Output layer:** a linear combination of those features.

```python
import numpy as np, os, json, textwrap
code = r'''
import numpy as np
def neighbors(p):
    r,c=divmod(p,9); out=[]
    for dr,dc in((-1,0),(1,0),(0,-1),(0,1)):
        rr,cc=r+dr,c+dc
        if 0<=rr<9 and 0<=cc<9: out.append(rr*9+cc)
    return out
NB=[neighbors(p) for p in range(81)]
PAR=[(p//9+p%9)%2 for p in range(81)]
def mdist(p,q): return abs(p//9-q//9)+abs(p%9-q%9)
def line(p):
    r,c=divmod(p,9); return min(r,c,8-r,8-c)
OWN,OPP,F0O,F1O,F0X,F1X=0,81,162,243,324,405
LIN,ESC,SA,EYE,INFP,INFN=0,81,162,243,324,405
DEFAULT=dict(K=5.0,B=1e4,C=5.0,W_cap=10.0,W_c=1.0,W_o=0.0,W_E=0.3,
             w_esc=3.0,w_sa=8.0,w_eye=25.0,alpha=3.0,t1=1.5,beta=1.0,t2=1.5,
             infw={1:1.0,2:0.5,3:0.25},prior={0:-1.5,1:-0.3,2:0.5,3:0.7,4:1.2},
             pass_logit=-5.0)
def build(P=None):
    Q=dict(DEFAULT); Q.update(P or {})
    K=Q['K']; B=Q['B']; T={}
    W=np.zeros((243,512),np.float32); b=np.zeros(512,np.float32)
    for p in range(81):
        W[3*p+0,OWN+p]=1; W[3*p+1,OPP+p]=1
        deg=len(NB[p]); aN=1-PAR[p]; aO=PAR[p]
        for plane,(b0,b1) in ((0,(F0O,F1O)),(1,(F0X,F1X))):
            base={0:b0,1:b1}
            for a,thr in ((aN,0),(aO,1)):
                u=base[a]+p
                for q in NB[p]:
                    W[3*q+0,u]-=1; W[3*q+1,u]-=1
                W[3*p+plane,u]+=K
                b[u]=deg-thr-K
    T['L1.W']=W; T['L1.b']=b
    for k in (2,3,4,5):
        W=np.zeros((512,512),np.float32); b=np.zeros(512,np.float32)
        for p in range(81):
            W[OWN+p,OWN+p]=1; W[OPP+p,OPP+p]=1
            for blk,g in ((F0O,OWN),(F1O,OWN),(F0X,OPP),(F1X,OPP)):
                s = 1.0 if k<5 else -1.0
                for q in NB[p]+[p]:
                    W[blk+q,blk+p]=s
                W[g+p,blk+p]=B; b[blk+p]=(-B if k<5 else 1-B)
        T[f'L{k}.W']=W; T[f'L{k}.b']=b
    W=np.zeros((512,512),np.float32); b=np.zeros(512,np.float32)
    iw=Q['infw']
    for e in range(81):
        deg=len(NB[e])
        u=LIN+e; b[u]=Q['C']+Q['W_E']*deg
        for n in NB[e]:
            W[F0X+n,u]+=Q['W_cap']; W[F1X+n,u]+=Q['W_cap']
            W[OPP+n,u]+=Q['W_c']-Q['W_E']; W[OWN+n,u]+=Q['W_o']-Q['W_E']
        u=ESC+e; b[u]=deg-4
        for n in NB[e]:
            W[OPP+n,u]-=1; W[F0O+n,u]+=2; W[F1O+n,u]+=2
        u=SA+e; b[u]=2-deg
        for n in NB[e]:
            W[OPP+n,u]+=1; W[OWN+n,u]-=1; W[F0O+n,u]+=2; W[F1O+n,u]+=2; W[F0X+n,u]-=2; W[F1X+n,u]-=2
        u=EYE+e; b[u]=1-deg
        for n in NB[e]: W[OWN+n,u]+=1
        b[INFP+e]=-Q['t1']; b[INFN+e]=-Q['t2']
        for q in range(81):
            d=mdist(e,q)
            if d in iw:
                W[OWN+q,INFP+e]+=iw[d]; W[OPP+q,INFP+e]-=iw[d]
                W[OWN+q,INFN+e]-=iw[d]; W[OPP+q,INFN+e]+=iw[d]
    T['L6.W']=W; T['L6.b']=b
    W=np.zeros((512,82),np.float32); b=np.zeros(82,np.float32)
    for e in range(81):
        W[LIN+e,e]=1; W[ESC+e,e]=Q['w_esc']; W[SA+e,e]=-Q['w_sa']; W[EYE+e,e]=-Q['w_eye']
        W[INFP+e,e]=-Q['alpha']; W[INFN+e,e]=-Q['beta']
        b[e]=Q['prior'][line(e)]-Q['C']
    b[81]=Q['pass_logit']
    T['out.W']=W; T['out.b']=b
    return T
'''
os.makedirs('/work',exist_ok=True)
open('/work/build.py','w').write(code)
exec(code)
T=build()
for k,v in T.items():
    assert tuple(v.shape)==tuple(SHAPES[k]) if isinstance(SHAPES,dict) else True, k
print(NAMES)
print(SHAPES)
for k,v in T.items(): save(k,v)
print(saved())
```