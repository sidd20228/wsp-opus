
import numpy as np
exec(open('/work/build.py').read().split('def build(')[0])
def gate_unit(L,Lp,u,p,sign,K):
    if sign>0: L.W[Lp.u('Z',p),u]+=K; L.b[u]+=-3*K
    else: L.W[Lp.u('Z',p),u]+=-K; L.b[u]+=K
def copy(L,Lp,name,src=None):
    for p in range(81): L.W[Lp.u(src or name,p),L.u(name,p)]=1
def build3(cfg, inp):
    OWN=lambda q: inp(q//9,q%9,0); OPP=lambda q: inp(q//9,q%9,1)
    d=cfg['delta']; K=8
    A=Lay(243)
    for nm in ['Z','G','EM','EO','Y']: A.add(nm)
    for p in range(81):
        n=nb(p); deg=len(n)
        u=A.u('Z',p); A.W[OWN(p),u]=1; A.W[OPP(p),u]=-1; A.b[u]=2
        # G = eps(p) if empty else 0
        u=A.u('G',p)
        for q in n: A.W[OWN(q),u]-=1; A.W[OPP(q),u]-=1
        A.W[OWN(p),u]-=K; A.W[OPP(p),u]-=K; A.b[u]=deg
        for nm,pl in (('EM',OWN),('EO',OPP)):
            u=A.u(nm,p)
            for q in n: A.W[OWN(q),u]-=1; A.W[OPP(q),u]-=1
            A.W[pl(p),u]+=K; A.b[u]=deg-K
        # Y = own isolated stone with e<=2 : relu(3 - e - K(1-own) - K*n_own)
        u=A.u('Y',p)
        for q in n: A.W[OWN(q),u]+=1; A.W[OPP(q),u]+=1; A.W[OWN(q),u]-=K
        A.W[OWN(p),u]+=K; A.b[u]=3-deg-K
    B=Lay(512)
    for nm in ['Z','WM2','HO','VO','W3M','YB']: B.add(nm)
    for p in range(81):
        u=B.u('Z',p); B.W[A.u('Z',p),u]=1; B.W[A.u('G',p),u]=d
        for nm,seg in (('HO',hseg),('VO',vseg)):
            u=B.u(nm,p)
            for q in seg(p): B.W[A.u('EO',q),u]+=1
            B.b[u]+=-1; gate_unit(B,A,u,p,-1,16)
        for nm,E,thr,sg in (('WM2','EM',1,1),('W3M','EM',2,1)):
            u=B.u(nm,p)
            for q in [p]+nb(p): B.W[A.u(E,q),u]+=1
            B.b[u]+=-thr; gate_unit(B,A,u,p,sg,24)
        u=B.u('YB',p)
        for q in nb(p): B.W[A.u('Y',q),u]+=1
        B.W[A.u('G',p),u]=-8
    C=Lay(512)
    for nm in ['Z','FM','FO','F3M','YB']: C.add(nm)
    copy(C,B,'Z'); copy(C,B,'YB')
    flood(C,B,'FM',['WM2'],+1,128); flood(C,B,'FO',['HO','VO'],-1,128); flood(C,B,'F3M',['W3M'],+1,128)
    D=Lay(512)
    for nm in ['Z','FM','FO','F3M','EYE','YB']: D.add(nm)
    copy(D,C,'Z'); copy(D,C,'YB')
    for nm,sg in (('FM',1),('FO',-1),('F3M',1)): flood(D,C,nm,[nm],sg,512)
    for p in range(81):
        u=D.u('EYE',p); n=nb(p); g=dg(p); deg=len(n)
        beta=0.25 if deg==4 else 0.5
        for q in n: D.W[C.u('Z',q),u]+=2
        for q in g: D.W[C.u('Z',q),u]+=beta
        D.b[u]=-6*deg+0.5-2*beta*len(g)
    E=Lay(512)
    for nm in ['Z','FM','FO','SA','EY']: E.add(nm)
    copy(E,D,'Z')
    flood(E,D,'FM',['FM'],+1,4096); flood(E,D,'FO',['FO'],-1,4096)
    for p in range(81):
        u=E.u('SA',p)   # relu(2 - eps - 2*sum F3M2(nbrs)); eps=(Z-2)/d for empty
        kap=cfg['kap']
        for q in nb(p): E.W[D.u('F3M',q),u]-=(2+8*kap)
        E.W[D.u('YB',p),u]+=kap
        E.W[D.u('Z',p),u]+=-1.0/d; E.b[u]=2+2.0/d
        u=E.u('EY',p); E.W[D.u('EYE',p),u]=cfg['w_eye']
    F=Lay(512)
    for nm in ['AtO','AtM','PEN','IP','IN']: F.add(nm)
    kern={int(k):v for k,v in cfg['kern'].items()}; th=cfg['theta']
    for p in range(81):
        u=F.u('AtO',p)
        for q in [p]+nb(p): F.W[E.u('FO',q),u]-=1
        F.b[u]+=1; gate_unit(F,E,u,p,-1,4)
        u=F.u('AtM',p)
        for q in [p]+nb(p): F.W[E.u('FM',q),u]-=1
        F.b[u]+=1; gate_unit(F,E,u,p,+1,4)
        u=F.u('PEN',p); F.W[E.u('SA',p),u]=cfg['w_sa']; F.W[E.u('EY',p),u]=1
        up=F.u('IP',p); un=F.u('IN',p); s=0
        for q in range(81):
            k=kern.get(mdist(p,q),0)
            if k: F.W[E.u('Z',q),up]+=k; F.W[E.u('Z',q),un]-=k; s+=k
        F.b[up]=-2*s-th; F.b[un]=2*s-th
    O=np.zeros((512,82)); ob=np.zeros(82)
    ck={int(k):v for k,v in cfg['conv'].items()}; sp=cfg['spos']
    for p in range(81):
        for q in nb(p):
            O[F.u('AtO',q),p]+=cfg['w_cap']
            O[F.u('AtM',q),p]+=cfg['w_esc']
        O[F.u('PEN',p),p]-=1
        for q in range(81):
            dd=mdist(p,q)
            if dd in ck:
                O[F.u('IP',q),p]-=sp*ck[dd]*cfg['own_scale']
                O[F.u('IN',q),p]-=sp*ck[dd]*cfg['opp_scale']
        ob[p]=sp*cfg['bias'][line(p)]
    ob[81]=cfg['P0']
    layers=[A,B,C,D,E,F]; out={}
    for i,Lx in enumerate(layers):
        out['L%d.W'%(i+1)]=Lx.W.astype(np.float32); out['L%d.b'%(i+1)]=Lx.b.astype(np.float32)
    out['out.W']=O.astype(np.float32); out['out.b']=ob.astype(np.float32)
    return out,[dict(L.ch) for L in layers]
