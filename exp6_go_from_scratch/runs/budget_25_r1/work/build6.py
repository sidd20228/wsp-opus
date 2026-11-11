
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
# L1-L4 layout
CC,F0O,F1O,GO,F0X,F1X=0,81,162,243,324,405
# L5 layout
OWN,OPP,AO,S3,NX0,NX1=0,81,162,243,324,405
# L6 layout
LIN,SA,SA2,NE,INFP,N5=0,81,162,243,324,405
DEFAULT=dict(K=5.0,B=1e4,C=10.0,W_cap=12.0,W_c=1.5,W_o=0.0,W_E=0.5,
             W_esc=6.0,w_sa=6.0,w_ne=25.0,w_sa2=5.0,alpha=1.0,t1=2.0,beta=1.0,t2=1.5,
             infw={1:1.0,2:0.5,3:0.25},prior={0:-3.0,1:-1.0,2:1.0,3:1.4,4:2.4},
             pass_logit=-16.0,gamma=0.3,s0=8.0,s1=25.0,prior_late={0:-0.5,1:0.0,2:0.3,3:0.3,4:0.3}, lin_inf=0.3,
             sa=dict(k0=4.0,E=2.0,t=4.0,w=1.0,ax=4.0))
def build(P=None):
    Q=dict(DEFAULT); Q.update(P or {})
    K=Q['K']; B=Q['B']; T={}
    # ---------- L1
    W=np.zeros((243,512),np.float32); b=np.zeros(512,np.float32)
    for p in range(81):
        deg=len(NB[p]); pi=PAR[p]
        W[3*p+0,CC+p]=1; W[3*p+1,CC+p]=-1; b[CC+p]=1
        def nterm(u,thr,gate_plane):
            for q in NB[p]:
                W[3*q+0,u]-=1; W[3*q+1,u]-=1
            W[3*p+gate_plane,u]+=K
            b[u]=deg-thr-K
        # own: parity flags (lib of parity 1-pi) and G (>=2 libs at stone)
        nterm((F0O if (1-pi)==0 else F1O)+p,0,0)
        nterm(GO+p,1,0)
        # opp: cross-trick parity flags
        nterm((F0X if (1-pi)==0 else F1X)+p,0,1)
        nterm((F0X if pi==0 else F1X)+p,1,1)
    T['L1.W']=W; T['L1.b']=b
    # ---------- L2,L3 propagate; L4: own negated, opp propagate
    for k in (2,3,4):
        W=np.zeros((512,512),np.float32); b=np.zeros(512,np.float32)
        for p in range(81):
            W[CC+p,CC+p]=1
            for blk in (F0O,F1O,GO):
                if k<4:
                    for q in NB[p]+[p]: W[blk+q,blk+p]=1
                    W[CC+p,blk+p]=B; b[blk+p]=-2*B
                else:
                    for q in NB[p]+[p]: W[blk+q,blk+p]=-1
                    W[CC+p,blk+p]=B; b[blk+p]=1-2*B
            for blk in (F0X,F1X):
                for q in NB[p]+[p]: W[blk+q,blk+p]=1
                W[CC+p,blk+p]=-B; b[blk+p]=0
        T[f'L{k}.W']=W; T[f'L{k}.b']=b
    # ---------- L5
    W=np.zeros((512,512),np.float32); b=np.zeros(512,np.float32)
    for p in range(81):
        W[CC+p,OWN+p]=1; b[OWN+p]=-1
        W[CC+p,OPP+p]=-1; b[OPP+p]=1
        for blk in (F0O,F1O,GO): W[blk+p,AO+p]=1
        b[AO+p]=-1
        W[CC+p,S3+p]=1; b[S3+p]=-1
        for blk in (F0O,F1O,GO): W[blk+p,S3+p]=-1
        for blk,dst in ((F0X,NX0),(F1X,NX1)):
            for q in NB[p]+[p]: W[blk+q,dst+p]=-1
            W[CC+p,dst+p]=-B; b[dst+p]=1
    T['L5.W']=W; T['L5.b']=b
    # ---------- L6
    W=np.zeros((512,512),np.float32); b=np.zeros(512,np.float32)
    iw=Q['infw']; sa=Q['sa']
    for e in range(81):
        deg=len(NB[e])
        u=LIN+e; b[u]=Q['C']+Q['W_E']*deg
        for n in NB[e]:
            W[OPP+n,u]+=Q['W_c']-Q['W_E']; W[OWN+n,u]+=Q['W_o']-Q['W_E']
            W[AO+n,u]+=Q['W_esc']
        u=SA+e; k0,cE,ct,cw,cax=sa['k0'],sa['E'],sa['t'],sa['w'],sa['ax']
        b[u]=k0-cE*deg
        for n in NB[e]:
            W[OWN+n,u]+=cE-cw; W[OPP+n,u]+=cE
            W[S3+n,u]+=-ct+cw; W[AO+n,u]+=cw
            W[NX0+n,u]-=cax; W[NX1+n,u]-=cax
        u=NE+e; b[u]=2-deg
        for n in NB[e]: W[OWN+n,u]+=1; W[OPP+n,u]-=2
        b[INFP+e]=-Q['t1']
        for q in range(81):
            d=mdist(e,q)
            if d in iw:
                W[OWN+q,INFP+e]+=iw[d]; W[OPP+q,INFP+e]-=iw[d]
                W[OWN+q,LIN+e]+=Q['lin_inf']*iw[d]; W[OPP+q,LIN+e]-=Q['lin_inf']*iw[d]
    # SA2 = relu(6 s_o + 4 s_x - 2 a_o - 6 t - 4 deg)   (E=0 connection to non-certified chain)
    for e in range(81):
        deg=len(NB[e]); u=SA2+e; b[u]=-4*deg
        for n in NB[e]:
            W[OWN+n,u]+=6; W[OPP+n,u]+=4; W[AO+n,u]-=2; W[S3+n,u]-=6
    # 5th hop, only the class matching the stone's parity
    for p in range(81):
        src_=NX0 if PAR[p]==0 else NX1
        for q in NB[p]+[p]:
            W[OPP+q,N5+p]-=1; W[src_+q,N5+p]+=1
        for q in NB[p]:
            W[OPP+q,N5+p]-=Q['gamma']
        W[OPP+p,N5+p]+=B; b[N5+p]=1-B
    # global phase units: G1=relu(S-s0), G2=relu(S-s1), S = number of stones
    for p in range(81):
        W[OWN+p,486]=1; W[OPP+p,486]=1; W[OWN+p,487]=1; W[OPP+p,487]=1
    b[486]=-Q['s0']; b[487]=-Q['s1']
    T['L6.W']=W; T['L6.b']=b
    W=np.zeros((512,82),np.float32); b=np.zeros(82,np.float32)
    for e in range(81):
        W[LIN+e,e]=1; W[SA+e,e]=-Q['w_sa']; W[NE+e,e]=-Q['w_ne']; W[INFP+e,e]=-Q['alpha']
        W[SA2+e,e]=-Q['w_sa2']
        for n in NB[e]:
            W[N5+n,e]+=Q['W_cap']
        b[e]=Q['prior'][line(e)]-Q['C']
        dl=(Q['prior_late'][line(e)]-Q['prior'][line(e)])/(Q['s1']-Q['s0'])
        W[486,e]+=dl; W[487,e]-=dl
    b[81]=Q['pass_logit']
    T['out.W']=W; T['out.b']=b
    return T
