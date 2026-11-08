Two big problems found: (1) self-ataris into 2-liberty groups (causing the repetition losses), (2) weak/random positional play (opening moves on the edge with flat probabilities). I'll restructure: star-witness floods for "≥3 liberties" (own and opponent) to detect 2-liberty self-ataris, and an influence term whose penalties are spatially convolved in the output layer (prefer contested regions), with sharper priors.

```python
build2 = r'''
import numpy as np
exec(open('/work/build.py').read().split('def build(')[0])
def gate_unit(L,Lp,u,p,sign,K):
    if sign>0: L.W[Lp.u('Z',p),u]+=K; L.b[u]+=-3*K
    else: L.W[Lp.u('Z',p),u]+=-K; L.b[u]+=K
def build2(cfg, inp):
    OWN=lambda q: inp(q//9,q%9,0); OPP=lambda q: inp(q//9,q%9,1)
    d=cfg['delta']
    A=Lay(243)
    for nm in ['Z','G','EM','EO']: A.add(nm)
    K=8
    for p in range(81):
        n=nb(p); deg=len(n)
        u=A.u('Z',p); A.W[OWN(p),u]=1; A.W[OPP(p),u]=-1; A.b[u]=2
        u=A.u('G',p)
        for q in n: A.W[OWN(q),u]-=1; A.W[OPP(q),u]-=1
        A.W[OWN(p),u]-=K; A.W[OPP(p),u]-=K; A.b[u]=deg-1
        for nm,pl in (('EM',OWN),('EO',OPP)):
            u=A.u(nm,p)
            for q in n: A.W[OWN(q),u]-=1; A.W[OPP(q),u]-=1
            A.W[pl(p),u]+=K; A.b[u]=deg-K
    B=Lay(512)
    for nm in ['Z','WM2','HO','VO','W3O','W3M']: B.add(nm)
    for p in range(81):
        u=B.u('Z',p); B.W[A.u('Z',p),u]=1; B.W[A.u('G',p),u]=d
        for nm,seg in (('HO',hseg),('VO',vseg)):
            u=B.u(nm,p)
            for q in seg(p): B.W[A.u('EO',q),u]+=1
            B.b[u]+=-1; gate_unit(B,A,u,p,-1,16)
        for nm,E,thr,sg in (('WM2','EM',1,1),('W3M','EM',2,1),('W3O','EO',2,-1)):
            u=B.u(nm,p)
            for q in [p]+nb(p): B.W[A.u(E,q),u]+=1
            B.b[u]+=-thr; gate_unit(B,A,u,p,sg,24)
    C=Lay(512)
    for nm in ['Z','FM','FO','F3O','F3M']: C.add(nm)
    for p in range(81): C.W[B.u('Z',p),C.u('Z',p)]=1
    flood(C,B,'FM',['WM2'],+1,128); flood(C,B,'FO',['HO','VO'],-1,128)
    flood(C,B,'F3O',['W3O'],-1,128); flood(C,B,'F3M',['W3M'],+1,128)
    D=Lay(512)
    for nm in ['Z','FM','FO','F3O','F3M','EYE']: D.add(nm)
    for p in range(81): D.W[C.u('Z',p),D.u('Z',p)]=1
    for nm,sg in (('FM',1),('FO',-1),('F3O',-1),('F3M',1)): flood(D,C,nm,[nm],sg,512)
    for p in range(81):
        u=D.u('EYE',p); n=nb(p); g=dg(p); deg=len(n)
        beta=0.25 if deg==4 else 0.5
        for q in n: D.W[C.u('Z',q),u]+=2
        for q in g: D.W[C.u('Z',q),u]+=beta
        D.b[u]=-6*deg+0.5-2*beta*len(g)
    E=Lay(512)
    for nm in ['Z','FM','FO','L2O','SA','EYE']: E.add(nm)
    for p in range(81): E.W[D.u('Z',p),E.u('Z',p)]=1
    flood(E,D,'FM',['FM'],+1,4096); flood(E,D,'FO',['FO'],-1,4096)
    Ksa=2.0/d
    for p in range(81):
        u=E.u('L2O',p)
        for q in [p]+nb(p): E.W[D.u('F3O',q),u]-=1
        E.b[u]+=1; gate_unit(E,D,u,p,-1,4)
        u=E.u('SA',p)
        for q in nb(p): E.W[D.u('F3M',q),u]-=1
        E.W[D.u('Z',p),u]+=-Ksa; E.b[u]=1+2*Ksa
        E.W[D.u('EYE',p),E.u('EYE',p)]=1
    F=Lay(512)
    for nm in ['AtO','AtM','L2O','PEN','IP','IN']: F.add(nm)
    kern={int(k):v for k,v in cfg['kern'].items()}; th=cfg['theta']
    for p in range(81):
        u=F.u('AtO',p)
        for q in [p]+nb(p): F.W[E.u('FO',q),u]-=1
        F.b[u]+=1; gate_unit(F,E,u,p,-1,4)
        u=F.u('AtM',p)
        for q in [p]+nb(p): F.W[E.u('FM',q),u]-=1
        F.b[u]+=1; gate_unit(F,E,u,p,+1,4)
        F.W[E.u('L2O',p),F.u('L2O',p)]=1
        u=F.u('PEN',p); F.W[E.u('SA',p),u]=cfg['w_sa']; F.W[E.u('EYE',p),u]=cfg['w_eye']
        up=F.u('IP',p); un=F.u('IN',p); s=0
        for q in range(81):
            k=kern.get(mdist(p,q),0)
            if k: F.W[E.u('Z',q),up]+=k; F.W[E.u('Z',q),un]-=k; s+=k
        F.b[up]=-2*s-th; F.b[un]=2*s-th
    O=np.zeros((512,82)); ob=np.zeros(82)
    ck={int(k):v for k,v in cfg['conv'].items()}
    for p in range(81):
        for q in nb(p):
            O[F.u('AtO',q),p]+=cfg['w_cap']-cfg['w_at']
            O[F.u('AtM',q),p]+=cfg['w_esc']
            O[F.u('L2O',q),p]+=cfg['w_at']
        O[F.u('PEN',p),p]-=1
        for q in range(81):
            dd=mdist(p,q)
            if dd in ck:
                O[F.u('IP',q),p]-=ck[dd]*cfg['own_scale']
                O[F.u('IN',q),p]-=ck[dd]*cfg['opp_scale']
        ob[p]=cfg['bias'][line(p)]
    ob[81]=cfg['P0']
    layers=[A,B,C,D,E,F]; out={}
    for i,Lx in enumerate(layers):
        out['L%d.W'%(i+1)]=Lx.W.astype(np.float32); out['L%d.b'%(i+1)]=Lx.b.astype(np.float32)
    out['out.W']=O.astype(np.float32); out['out.b']=ob.astype(np.float32)
    return out,[dict(L.ch) for L in layers]
'''
open('/work/build2.py','w').write(build2)
exec(open('/work/build2.py').read())
import json
cfg=dict(delta=0.002, kern={0:1.0,1:1.0,2:0.5,3:0.25,4:0.125}, theta=0.75,
         conv={0:2.0,1:0.4,2:0.2,3:0.1}, own_scale=1.0, opp_scale=1.0,
         w_eye=25.0, w_cap=25.0, w_esc=7.0, w_at=3.0, w_sa=10.0, bias=[-3.0,-1.0,0.8,1.5,1.8], P0=-4.0)
arrs,layout=build2(cfg,inp)
for k,v in arrs.items(): save(k,v)
json.dump(cfg,open('/work/cfg_v2.json','w')); json.dump(layout,open('/work/layout_v2.json','w'))
print(layout)
print({k:float(np.abs(v).max()) for k,v in arrs.items()})
```