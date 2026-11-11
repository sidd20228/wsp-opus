
import numpy as np
exec(open('/work/go.py').read())
def build(P):
    W={n:np.zeros(s,np.float32) for n,s in SHAPES_.items()}
    rc=lambda p:(p//9,p%9)
    deg=[len(NB[p]) for p in range(81)]
    # ids
    ido=np.array([(r+5*c)%18 for r in range(9) for c in range(9)],float)
    idm=np.random.default_rng(12345).permutation(81).astype(float)
    OWN=lambda p:3*p; OPP=lambda p:3*p+1
    # layout L1..L4
    V=0; OY=81; OX=162; OQ=243; MY=324; MX=405; PR=486
    L1=W['L1.W']; b1=W['L1.b']
    for p in range(81):
        # V = 1 - own + opp
        L1[OWN(p),V+p]=-1; L1[OPP(p),V+p]=1; b1[V+p]=1
        # channels
        for (ch,idv,gate,G) in ((OY,None,OPP,4),(OX,ido,OPP,4*17+1),(OQ,ido**2,OPP,4*289+1),
                                (MY,None,OWN,4),(MX,idm,OWN,4*80+1)):
            tot=0
            for q in NB[p]:
                w=1.0 if idv is None else idv[q]
                L1[OWN(q),ch+p]+=-w; L1[OPP(q),ch+p]+=-w; tot+=w
            L1[gate(p),ch+p]+=G; b1[ch+p]=tot-G
    # probes
    L1[OWN(40),PR]=16777000; b1[PR]=1
    L1[OWN(40),PR+1]=1
    L1[OWN(40),PR+2]=2049
    L1[OWN(40),PR+3]=1.0001
    L1max={OY:4,OX:68,OQ:1156,MY:4,MX:320}
    for k in (2,3,4):
        Wk=W[f'L{k}.W']; bk=W[f'L{k}.b']
        for p in range(81):
            Wk[V+p,V+p]=1
            for ch in (OY,OX,OQ,MY,MX):
                M=4*L1max[ch]*5**(k-2)+1
                Wk[ch+p,ch+p]=1
                for q in NB[p]: Wk[ch+q,ch+p]=1
                if ch in (OY,OX,OQ):
                    Wk[V+p,ch+p]=M; bk[ch+p]=-2*M
                else:
                    Wk[V+p,ch+p]=-M
        if k==2:
            Wk[PR,PR]=1; Wk[PR+1,PR]=-16777000
        else:
            Wk[PR,PR]=1
        for j in (1,2,3): Wk[PR+j,PR+j]=1
    # L5
    V5=0; O5=81; CE=162; AM=306; BM=387
    EDGES=[(a,b) for a in range(81) for b in NB[a] if a<b]
    L5=W['L5.W']; b5=W['L5.b']
    for p in range(81):
        L5[V+p,V5+p]=1
        L5[V+p,O5+p]=1; b5[O5+p]=-1
        cm=idm[p]
        for q in NB[p]:
            L5[MX+q,AM+p]+=1; L5[MY+q,AM+p]+=-cm
            L5[MX+q,BM+p]+=-1; L5[MY+q,BM+p]+=cm
    for e,(a,bb) in enumerate(EDGES):
        u=CE+e; b5[u]=1
        for s,t in ((a,bb),(bb,a)):
            c=ido[t]
            L5[OQ+s,u]+=-1; L5[OX+s,u]+=2*c; L5[OY+s,u]+=-c*c
        M=2.0
        L5[V+a,u]+=M; L5[V+bb,u]+=M; b5[u]+=-3*M
    for j in range(4): L5[PR+j,486+j]=1
    # L6
    CAP=0; OA=81; SA=162; EYE=243; TA=324; TB=405
    L6=W['L6.W']; b6=W['L6.b']
    K=P['K']
    for p in range(81):
        def add_own(q,unit,w):  # own_q = 1 - V + O
            L6[V5+q,unit]+=-w; L6[O5+q,unit]+=w; b6[unit]+=w
        def add_emp(q,unit,w):  # empty = V - 2O
            L6[V5+q,unit]+=w; L6[O5+q,unit]+=-2*w
        def add_opp(q,unit,w):
            L6[O5+q,unit]+=w
        # CAP = relu(sum of capture-edge flags at p)
        for e,(a,bb) in enumerate(EDGES):
            if p in (a,bb): L6[CE+e,CAP+p]=1
        cesum=[e for e,(a,bb) in enumerate(EDGES) if p in (a,bb)]
        if P.get('mode','p3')=='p3':
            # OA = relu(nown - 5(AM+BM))
            L6[AM+p,OA+p]=-5; L6[BM+p,OA+p]=-5
            for q in NB[p]: add_own(q,OA+p,1)
        else:
            # P1u = relu(2 - nown - 3E - 9*sumCE)   (E0 & nown<=1, non-capture)
            b6[OA+p]+=2
            for q in NB[p]: add_own(q,OA+p,-1); add_emp(q,OA+p,-3)
            for e in cesum: L6[CE+e,OA+p]+=-9
        # SA = relu(2 - E - 3(AM+BM))
        b6[SA+p]+=2; L6[AM+p,SA+p]=-3; L6[BM+p,SA+p]=-3
        for q in NB[p]: add_emp(q,SA+p,-1)
        # EYE = relu(nown - deg + 1)
        b6[EYE+p]+=1-deg[p]
        for q in NB[p]: add_own(q,EYE+p,1)
        # tents: infl = sum K (1 - V)
        r,cc=rc(p)
        for q in range(81):
            r2,c2=rc(q); d2=(r-r2)**2+(cc-c2)**2
            if d2==0 or d2 not in K: continue
            k_=K[d2]
            for u in (TA+p,TB+p):
                b6[u]+=-k_; L6[V5+q,u]+=k_   # -infl
        Mt=P['Mt']
        for u,t in ((TA+p,P['T']),(TB+p,0.0)):
            b6[u]+=t
            # -Mt*(1-empty) = -Mt + Mt*(V-2O)
            b6[u]+=-Mt; L6[V5+p,u]+=Mt; L6[O5+p,u]+=-2*Mt
    for j in range(4): L6[486+j,486+j]=1
    # output
    Wo=W['out.W']; bo=W['out.b']
    for p in range(81):
        r,cc=rc(p)
        line=min(r,cc,8-r,8-cc)
        bo[p]=P['bias']+P['linebias'][line]
        Wo[CAP+p,p]=P['wcap']; Wo[OA+p,p]=(P['woa'] if P.get('mode','p3')=='p3' else -P['wp1']); Wo[SA+p,p]=-P['wsa']; Wo[EYE+p,p]=-P['weye']
        for q in range(81):
            r2,c2=rc(q); d2=(r-r2)**2+(cc-c2)**2
            if d2 in P['K2']:
                Wo[TA+q,p]+=P['alpha']*P['K2'][d2]; Wo[TB+q,p]+=-P['gamma']*P['alpha']*P['K2'][d2]
    bo[81]=P['bpass']
    return W
P0=dict(K={1:1.0,2:0.65,4:0.45,5:0.3,8:0.15,9:0.15,10:0.1},T=2.5,Mt=60.0,
        bias=-3.0,linebias={0:-1.5,1:-0.3,2:0.0,3:0.0,4:0.0},
        wcap=10.0,woa=8.0,wsa=12.0,weye=20.0,alpha=3.6,K2={0:0.6,1:0.1},bpass=0.0)
