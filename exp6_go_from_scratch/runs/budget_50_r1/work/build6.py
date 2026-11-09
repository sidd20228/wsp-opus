
import numpy as np
exec(open('/work/build.py').read().split('def build(')[0])
def gate_unit(L,Lp,u,p,sign,K):
    if sign>0: L.W[Lp.u('Z',p),u]+=K; L.b[u]+=-3*K
    else: L.W[Lp.u('Z',p),u]+=-K; L.b[u]+=K
def copy(L,Lp,name,src=None):
    for p in range(81): L.W[Lp.u(src or name,p),L.u(name,p)]=1
def build6(cfg, inp):
    OWN=lambda q: inp(q//9,q%9,0); OPP=lambda q: inp(q//9,q%9,1)
    d=cfg['delta']; K=8
    A=Lay(243)
    for nm in ['Z','G','EM','EO','Y']: A.add(nm)
    for p in range(81):
        n=nb(p); deg=len(n)
        u=A.u('Z',p); A.W[OWN(p),u]=1; A.W[OPP(p),u]=-1; A.b[u]=2
        u=A.u('G',p)
        for q in n: A.W[OWN(q),u]-=1; A.W[OPP(q),u]-=1
        A.W[OWN(p),u]-=K; A.W[OPP(p),u]-=K; A.b[u]=deg
        for nm,pl in (('EM',OWN),('EO',OPP)):
            u=A.u(nm,p)
            for q in n: A.W[OWN(q),u]-=1; A.W[OPP(q),u]-=1
            A.W[pl(p),u]+=K; A.b[u]=deg-K
        u=A.u('Y',p)
        iso=cfg.get('y_iso',1.0)
        for q in n: A.W[OWN(q),u]+=1-K*iso; A.W[OPP(q),u]+=1
        A.W[OWN(p),u]+=K; A.b[u]=3-deg-K
    B=Lay(512)
    for nm in ['Z','HO','VO','W3O','W3M','YB']: B.add(nm)
    for p in range(81):
        u=B.u('Z',p); B.W[A.u('Z',p),u]=1; B.W[A.u('G',p),u]=d
        for nm,seg in (('HO',hseg),('VO',vseg)):
            u=B.u(nm,p)
            for q in seg(p): B.W[A.u('EO',q),u]+=1
            B.b[u]+=-1; gate_unit(B,A,u,p,-1,16)
        for nm,E,thr,sg in (('W3O','EO',2,-1),('W3M','EM',2,1)):
            u=B.u(nm,p)
            for q in [p]+nb(p): B.W[A.u(E,q),u]+=1
            B.b[u]+=-thr; gate_unit(B,A,u,p,sg,24)
        u=B.u('YB',p)
        for q in nb(p): B.W[A.u('Y',q),u]+=1
        B.W[A.u('G',p),u]=-16
    C=Lay(512)
    for nm in ['Z','FO','F3O','F3M','YB']: C.add(nm)
    copy(C,B,'Z'); copy(C,B,'YB')
    flood(C,B,'FO',['HO','VO'],-1,128); flood(C,B,'F3O',['W3O'],-1,128); flood(C,B,'F3M',['W3M'],+1,128)
    D=Lay(512)
    for nm in ['Z','FO','F3O','F3M','EYE','YB']: D.add(nm)
    copy(D,C,'Z'); copy(D,C,'YB')
    for nm,sg in (('FO',-1),('F3O',-1),('F3M',1)): flood(D,C,nm,[nm],sg,512)
    for p in range(81):
        u=D.u('EYE',p); n=nb(p); g=dg(p); deg=len(n)
        a_,b_={4:(0.3,0.0),3:(0.75,-0.5),2:(0.5,0.5)}[deg]
        for q in n: D.W[C.u('Z',q),u]+=2
        for q in g: D.W[C.u('Z',q),u]+=a_
        D.b[u]=-6*deg+b_-2*a_*len(g)
    E=Lay(512)
    for nm in ['Z','FO','L2O','L2M','SA','EY']: E.add(nm)
    copy(E,D,'Z')
    kap=cfg['kap']
    flood(E,D,'FO',['FO'],-1,4096)
    for p in range(81):
        for nm,fl,sg in (('L2O','F3O',-1),('L2M','F3M',1)):
            u=E.u(nm,p)
            for q in [p]+nb(p): E.W[D.u(fl,q),u]-=1
            E.b[u]+=1; gate_unit(E,D,u,p,sg,4)
        u=E.u('SA',p)
        for q in nb(p): E.W[D.u('F3M',q),u]-=(2+16*kap)
        E.W[D.u('YB',p),u]+=kap
        E.W[D.u('Z',p),u]+=-1.0/d; E.b[u]=2+2.0/d
        u=E.u('EY',p); E.W[D.u('EYE',p),u]=cfg['w_eye']
    F=Lay(512)
    for nm in ['TAC','PEN','BA','BB','BC','AtO']: F.add(nm)
    kern={int(k):v for k,v in cfg['kern'].items()}; th=cfg['theta']
    for p in range(81):
        u=F.u('TAC',p)
        for q in nb(p):
            F.W[E.u('L2O',q),u]+=cfg['w_at']
            F.W[E.u('L2M',q),u]+=cfg['w_esc']
        u=F.u('PEN',p); F.W[E.u('SA',p),u]=cfg['w_sa']; F.W[E.u('EY',p),u]=1
        u=F.u('AtO',p)
        for q in [p]+nb(p): F.W[E.u('FO',q),u]-=1
        F.b[u]+=1; gate_unit(F,E,u,p,-1,4)
        ua=F.u('BA',p); ub=F.u('BB',p); uc=F.u('BC',p); s=0
        om=cfg.get('omega',0.0)
        Mp=sum(kern.get(mdist(p,q),0) for q in range(81)); Mc=sum(kern.get(mdist(40,q),0) for q in range(81))
        fn=(Mc/Mp)**cfg.get('gnorm',0.0)
        for q in range(81):
            k=kern.get(mdist(p,q),0)*fn
            if k:
                for uu in (ua,ub,uc):
                    F.W[E.u('Z',q),uu]+=k
                    F.W[E.u('L2M',q),uu]-=k*om
                    F.W[E.u('L2O',q),uu]+=k*om
                s+=k
        F.b[ua]=-2*s+th; F.b[ub]=-2*s; F.b[uc]=-2*s-th
    O=np.zeros((512,82)); ob=np.zeros(82)
    ck={int(k):v for k,v in cfg['conv'].items()}; sp=cfg['spos']
    for p in range(81):
        O[F.u('TAC',p),p]+=1
        for q in nb(p): O[F.u('AtO',q),p]+=cfg['w_cap']-cfg['w_at']
        O[F.u('PEN',p),p]-=1
        for q in range(81):
            dd=mdist(p,q)
            if dd in ck:
                w=sp*cfg['alpha']*ck[dd]
                O[F.u('BA',q),p]+=w; O[F.u('BB',q),p]-=2*w; O[F.u('BC',q),p]+=w
        O[F.u('BC',p),p]-=sp*cfg['beta_own']
        ob[p]=sp*cfg['bias'][line(p)]
    ob[81]=cfg['P0']
    lam=cfg.get('lam',1.0); O*=lam; ob*=lam
    layers=[A,B,C,D,E,F]; out={}
    for i,Lx in enumerate(layers):
        out['L%d.W'%(i+1)]=Lx.W.astype(np.float32); out['L%d.b'%(i+1)]=Lx.b.astype(np.float32)
    out['out.W']=O.astype(np.float32); out['out.b']=ob.astype(np.float32)
    return out,[dict(L.ch) for L in layers]
