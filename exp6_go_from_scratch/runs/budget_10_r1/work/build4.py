
import numpy as np
exec(open('/work/go.py').read())
def build4(P):
    W={n:np.zeros(s,np.float32) for n,s in SHAPES_.items()}
    rc=lambda p:(p//9,p%9)
    deg=[len(NB[p]) for p in range(81)]
    ido=np.array([(r+7*c)%25 for r in range(9) for c in range(9)],float)
    OWN=lambda p:3*p; OPP=lambda p:3*p+1
    V=0; OY=81; OX=162; OQ=243; TA=324; TB=405; G=324; O4=405
    L1=W['L1.W']; b1=W['L1.b']; K=P['K']
    for p in range(81):
        L1[OWN(p),V+p]=-1; L1[OPP(p),V+p]=1; b1[V+p]=1
        for (ch,idv,G1) in ((OY,None,4),(OX,ido,4*24+1),(OQ,ido**2,4*576+1)):
            tot=0
            for q in NB[p]:
                w=1.0 if idv is None else idv[q]
                L1[OWN(q),ch+p]+=-w; L1[OPP(q),ch+p]+=-w; tot+=w
            L1[OPP(p),ch+p]+=G1; b1[ch+p]=tot-G1
        r,c=rc(p); Mt=P['Mt']
        for u,t in ((TA+p,P['T']),(TB+p,0.0)):
            for q in range(81):
                r2,c2=rc(q); d2=(r-r2)**2+(c-c2)**2
                if d2==0 or d2 not in K: continue
                L1[OWN(q),u]+=-K[d2]; L1[OPP(q),u]+=K[d2]
            b1[u]+=t; L1[OWN(p),u]+=-Mt; L1[OPP(p),u]+=-Mt
    L1max={OY:4,OX:96,OQ:2304}; C=P['C']
    for k in (2,3,4,5):
        Wk=W[f'L{k}.W']; bk=W[f'L{k}.b']
        for p in range(81):
            if k<5: Wk[V+p,V+p]=1
            for ch in (OY,OX,OQ):
                M=4*L1max[ch]*5**(k-2)+1
                Wk[ch+p,ch+p]=1
                for q in NB[p]: Wk[ch+q,ch+p]=1
                Wk[V+p,ch+p]=M; bk[ch+p]=-2*M
            if k==2:
                r,c=rc(p)
                for q in range(81):
                    r2,c2=rc(q); d2=(r-r2)**2+(c-c2)**2
                    if d2 in P['K2']:
                        Wk[TA+q,G+p]+=P['alpha']*P['K2'][d2]; Wk[TB+q,G+p]+=-P['gamma']*P['alpha']*P['K2'][d2]
                bk[G+p]=C
            elif k in (3,4):
                Wk[G+p,G+p]=1
            if k==4:   # opp indicator at L4
                Wk[V+p,O4+p]=1; bk[O4+p]=-1
    # L5: V' = V4 + eps*G4 ; O5 ; LS5
    eps=P['eps']; O5=324; LS=405
    L5=W['L5.W']; b5=W['L5.b']
    for p in range(81):
        L5[V+p,V+p]=1; L5[G+p,V+p]=eps
        L5[V+p,O5+p]=1; b5[O5+p]=-1
        # LS = relu(4own_s - 4 + nopp - deg + 3 - 4nown); own = 1 - V + O
        u=LS+p
        L5[V+p,u]+=-4; L5[O4+p,u]+=4; b5[u]+=4-4+3-deg[p]
        for q in NB[p]:
            L5[O4+q,u]+=1
            L5[V+q,u]+=4; L5[O4+q,u]+=-4; b5[u]+=-4
    EDGES=[(a,b) for a in range(81) for b in NB[a] if a<b]
    CE=0; SA=144; Z1=225; EYE=306; G6=387
    L6=W['L6.W']; b6=W['L6.b']
    for e,(a,bb) in enumerate(EDGES):
        u=CE+e; b6[u]=1
        for s,t in ((a,bb),(bb,a)):
            c=ido[t]
            L6[OQ+s,u]+=-2; L6[OX+s,u]+=4*c; L6[OY+s,u]+=-2*c*c
        Mg=2.0
        L6[V+a,u]+=Mg; L6[V+bb,u]+=Mg; b6[u]+=-3*Mg
    kap=P['kappa']
    for p in range(81):
        L6[V+p,G6+p]=1
        def add_own(q,unit,w): L6[V+q,unit]+=-w; L6[O5+q,unit]+=w; b6[unit]+=w
        def add_emp(q,unit,w): L6[V+q,unit]+=w; L6[O5+q,unit]+=-2*w
        b6[SA+p]+=2+kap*(1-2*deg[p])
        for q in NB[p]: add_emp(q,SA+p,-1); L6[V+q,SA+p]+=kap
        # Z1' = relu(nown - 8E + 3 + 6*sumLS)
        b6[Z1+p]+=3
        for q in NB[p]: add_own(q,Z1+p,1); add_emp(q,Z1+p,-8); L6[LS+q,Z1+p]+=P['lsw']
        b6[EYE+p]+=1-deg[p]
        for q in NB[p]: add_own(q,EYE+p,1)
    Wo=W['out.W']; bo=W['out.b']
    for p in range(81):
        r,c=rc(p); line=min(r,c,8-r,8-c)
        bo[p]=P['bias']+P['linebias'][line]-C-1.0/eps
        Wo[G6+p,p]=1.0/eps
        Wo[SA+p,p]=-P['wsa']; Wo[Z1+p,p]=-P['wz1']; Wo[EYE+p,p]=-P['weye']
    for e,(a,bb) in enumerate(EDGES):
        Wo[CE+e,a]+=P['wcap']; Wo[CE+e,bb]+=P['wcap']
    bo[81]=P['bpass']
    return W
PV4=dict(K={1:1.0,2:0.65,4:0.45,5:0.3,8:0.15,9:0.15,10:0.1},T=2.5,Mt=60.0,C=100.0,eps=2.0**-14,
        bias=-2.0,linebias={0:-2.5,1:-1.0,2:0.0,3:0.0,4:0.2},alpha=5.0,gamma=1.5,K2={0:0.6,1:0.1},
        wcap=25.0,wsa=8.0,wz1=3.0,weye=15.0,kappa=2.0,lsw=6.0,bpass=-12.0)
