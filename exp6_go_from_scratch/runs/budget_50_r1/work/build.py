
import numpy as np
def nb(p):
    r,c=divmod(p,9); o=[]
    for dr,dc in ((-1,0),(1,0),(0,-1),(0,1)):
        rr,cc=r+dr,c+dc
        if 0<=rr<9 and 0<=cc<9: o.append(rr*9+cc)
    return o
def dg(p):
    r,c=divmod(p,9); o=[]
    for dr,dc in ((-1,-1),(-1,1),(1,-1),(1,1)):
        rr,cc=r+dr,c+dc
        if 0<=rr<9 and 0<=cc<9: o.append(rr*9+cc)
    return o
def hseg(p):
    r,c=divmod(p,9); return [r*9+cc for cc in (c-1,c,c+1) if 0<=cc<9]
def vseg(p):
    r,c=divmod(p,9); return [rr*9+c for rr in (r-1,r,r+1) if 0<=rr<9]
def mdist(p,q): return abs(p//9-q//9)+abs(p%9-q%9)
def line(p):
    r,c=divmod(p,9); return min(r,8-r,c,8-c)

class Lay:
    def __init__(s,nin):
        s.W=np.zeros((nin,512)); s.b=np.zeros(512); s.n=0; s.ch={}
    def add(s,name,k=81):
        s.ch[name]=s.n; s.n+=k
        assert s.n<=512,(name,s.n)
        return s.ch[name]
    def u(s,name,p=0): return s.ch[name]+p

def flood(Lc,Lp,name,src,sign,K):
    # Lc current layer, Lp previous; src: list of channel names in Lp summed over {p}+nb(p)
    for p in range(81):
        u=Lc.u(name,p)
        for q in [p]+nb(p):
            for s in src: Lc.W[Lp.u(s,q),u]+=1
        if sign>0: Lc.W[Lp.u('Z',p),u]+=K; Lc.b[u]+=-3*K
        else: Lc.W[Lp.u('Z',p),u]+=-K; Lc.b[u]+=K

def build(cfg, inp):
    OWN=lambda q: inp(q//9,q%9,0)
    OPP=lambda q: inp(q//9,q%9,1)
    WTM=lambda q: inp(q//9,q%9,2)
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
    for nm in ['Z','HM','VM','HO','VO','W3O']: B.add(nm)
    K2=16
    for p in range(81):
        u=B.u('Z',p); B.W[A.u('Z',p),u]=1; B.W[A.u('G',p),u]=d
        for nm,seg in (('HM',hseg),('VM',vseg)):
            u=B.u(nm,p)
            for q in seg(p): B.W[A.u('EM',q),u]+=1
            B.W[A.u('Z',p),u]+=K2; B.b[u]=-1-3*K2
        for nm,seg in (('HO',hseg),('VO',vseg)):
            u=B.u(nm,p)
            for q in seg(p): B.W[A.u('EO',q),u]+=1
            B.W[A.u('Z',p),u]+=-K2; B.b[u]=-1+K2
        u=B.u('W3O',p)
        for q in [p]+nb(p): B.W[A.u('EO',q),u]+=1
        B.W[A.u('Z',p),u]+=-24; B.b[u]=-2+24
    C=Lay(512)
    for nm in ['Z','FM','FO','F3O','IP','IN']: C.add(nm)
    for p in range(81): C.W[B.u('Z',p),C.u('Z',p)]=1
    flood(C,B,'FM',['HM','VM'],+1,128)
    flood(C,B,'FO',['HO','VO'],-1,128)
    flood(C,B,'F3O',['W3O'],-1,128)
    kern={int(k):v for k,v in cfg['kern'].items()}; th=cfg['theta']
    for p in range(81):
        up=C.u('IP',p); un=C.u('IN',p); s=0
        for q in range(81):
            k=kern.get(mdist(p,q),0)
            if k: C.W[B.u('Z',q),up]+=k; C.W[B.u('Z',q),un]-=k; s+=k
        C.b[up]=-2*s-th; C.b[un]=2*s-th
    D=Lay(512)
    for nm in ['Z','FM','FO','F3O','PL','EYE']: D.add(nm)
    for p in range(81): D.W[C.u('Z',p),D.u('Z',p)]=1
    flood(D,C,'FM',['FM'],+1,512)
    flood(D,C,'FO',['FO'],-1,512)
    flood(D,C,'F3O',['F3O'],-1,512)
    al=cfg['alpha']; C4=cfg['C4']
    for p in range(81):
        u=D.u('PL',p); D.W[C.u('IP',p),u]=-al; D.W[C.u('IN',p),u]=-al; D.b[u]=C4
        u=D.u('EYE',p); n=nb(p); g=dg(p); deg=len(n)
        beta=0.25 if deg==4 else 0.5
        for q in n: D.W[C.u('Z',q),u]+=2
        for q in g: D.W[C.u('Z',q),u]+=beta
        D.b[u]=-6*deg+0.5-2*beta*len(g)
    E=Lay(512)
    for nm in ['Z','FM','FO','F3O','PL']: E.add(nm)
    for p in range(81): E.W[D.u('Z',p),E.u('Z',p)]=1
    flood(E,D,'FM',['FM'],+1,4096)
    flood(E,D,'FO',['FO'],-1,4096)
    flood(E,D,'F3O',['F3O'],-1,4096)
    C5=cfg['C5']
    for p in range(81):
        u=E.u('PL',p); E.W[D.u('PL',p),u]=1; E.W[D.u('EYE',p),u]=-cfg['w_eye']; E.b[u]=C5
    F=Lay(512)
    for nm in ['AtO','AtM','L2O','SA','PL']: F.add(nm)
    K6=4; Ksa=2.0/d
    for p in range(81):
        u=F.u('AtO',p)
        for q in [p]+nb(p): F.W[E.u('FO',q),u]-=1
        F.W[E.u('Z',p),u]+=-K6; F.b[u]=1+K6
        u=F.u('AtM',p)
        for q in [p]+nb(p): F.W[E.u('FM',q),u]-=1
        F.W[E.u('Z',p),u]+=K6; F.b[u]=1-3*K6
        u=F.u('L2O',p)
        for q in [p]+nb(p): F.W[E.u('F3O',q),u]-=1
        F.W[E.u('Z',p),u]+=-K6; F.b[u]=1+K6
        u=F.u('SA',p)
        for q in nb(p): F.W[E.u('FM',q),u]-=1
        F.W[E.u('Z',p),u]+=-Ksa; F.b[u]=1+2*Ksa
        u=F.u('PL',p); F.W[E.u('PL',p),u]=1
    O=np.zeros((512,82)); ob=np.zeros(82)
    for p in range(81):
        O[F.u('PL',p),p]=1
        for q in nb(p):
            O[F.u('AtO',q),p]+=cfg['w_cap']-cfg['w_at']
            O[F.u('AtM',q),p]+=cfg['w_esc']
            O[F.u('L2O',q),p]+=cfg['w_at']
        O[F.u('SA',p),p]-=cfg['w_sa']
        ob[p]=cfg['bias'][line(p)]-C4-C5
    ob[81]=cfg['P0']
    layers=[A,B,C,D,E,F]
    out={}
    for i,Lx in enumerate(layers):
        out['L%d.W'%(i+1)]=Lx.W.astype(np.float32); out['L%d.b'%(i+1)]=Lx.b.astype(np.float32)
    out['out.W']=O.astype(np.float32); out['out.b']=ob.astype(np.float32)
    layout=[dict(L.ch) for L in layers]
    return out, layout
