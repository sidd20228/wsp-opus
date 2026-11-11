
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
CC,F0O,F1O,GO,F0X,F1X=0,81,162,243,324,405          # L1-L4
OWN,OPP,AO,S3,NX0,NX1=0,81,162,243,324,405          # L5
LIN,SA,NE,INFP,N5,SAC=0,81,162,243,324,405           # L6
G1,G2=486,487
DEFAULT=dict(K=5.0,B=1e4,C=10.0,W_cap=12.0,W_c=1.5,W_o=0.0,W_E=0.5,W_esc=6.0,
             w_sa=6.0,w_ne=25.0,w_ko=30.0,alpha=3.0,t1=1.5,lin_inf=0.3,
             infw={1:1.0,2:0.5,3:0.25},
             prior={0:-3.0,1:-1.0,2:1.0,3:1.4,4:2.4},
             prior_late={0:-1.0,1:-0.3,2:0.5,3:0.7,4:1.2},s0=40.0,s1=70.0,
             pass_logit=-5.0,gamma=0.3,
             lone=dict(k0=10.0,E=8.0,ax=5.0,mu=10.0),conn=dict(ao=4.0,w=3.0,E=2.0,t=4.0,ax=1.0),w_conn=6.0)
def build(P=None):
    Q=dict(DEFAULT); Q.update(P or {})
    K=Q['K']; B=Q['B']; T={}
    W=np.zeros((243,512),np.float32); b=np.zeros(512,np.float32)
    for p in range(81):
        deg=len(NB[p]); pi=PAR[p]
        W[3*p+0,CC+p]=1; W[3*p+1,CC+p]=-1; b[CC+p]=1
        def nterm(u,thr,gate_plane):
            for q in NB[p]:
                W[3*q+0,u]-=1; W[3*q+1,u]-=1
            W[3*p+gate_plane,u]+=K
            b[u]=deg-thr-K
        nterm((F0O if (1-pi)==0 else F1O)+p,0,0)
        nterm(GO+p,1,0)
        nterm((F0X if (1-pi)==0 else F1X)+p,0,1)
        nterm((F0X if pi==0 else F1X)+p,1,1)
    T['L1.W']=W; T['L1.b']=b
    for k in (2,3,4):
        W=np.zeros((512,512),np.float32); b=np.zeros(512,np.float32)
        for p in range(81):
            W[CC+p,CC+p]=1
            for blk in (F0O,F1O,GO):
                s=1 if k<4 else -1
                for q in NB[p]+[p]: W[blk+q,blk+p]=s
                W[CC+p,blk+p]=B; b[blk+p]=(-2*B if k<4 else 1-2*B)
            for blk in (F0X,F1X):
                for q in NB[p]+[p]: W[blk+q,blk+p]=1
                W[CC+p,blk+p]=-B; b[blk+p]=0
        T[f'L{k}.W']=W; T[f'L{k}.b']=b
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
    W=np.zeros((512,512),np.float32); b=np.zeros(512,np.float32)
    iw=Q['infw']
    for e in range(81):
        deg=len(NB[e])
        u=LIN+e; b[u]=Q['C']+Q['W_E']*deg
        for n in NB[e]:
            W[OPP+n,u]+=Q['W_c']-Q['W_E']; W[OWN+n,u]+=Q['W_o']-Q['W_E']
            W[AO+n,u]+=Q['W_esc']
        # SA_lone = relu(k0 - cE*E - cax*a_x - mu*s_o)   (E = deg - s_o - s_x)
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
        u=NE+e; b[u]=2-deg
        for n in NB[e]: W[OWN+n,u]+=1; W[OPP+n,u]-=2; W[AO+n,u]-=2
        b[INFP+e]=-Q['t1']
        for q in range(81):
            d=mdist(e,q)
            if d in iw:
                W[OWN+q,INFP+e]+=iw[d]; W[OPP+q,INFP+e]-=iw[d]
                W[OWN+q,LIN+e]+=Q['lin_inf']*iw[d]; W[OPP+q,LIN+e]-=Q['lin_inf']*iw[d]
    for p in range(81):
        src_=NX0 if PAR[p]==0 else NX1
        for q in NB[p]+[p]:
            W[OPP+q,N5+p]-=1; W[src_+q,N5+p]+=1
        for q in NB[p]:
            W[OPP+q,N5+p]-=Q['gamma']
        W[OPP+p,N5+p]+=B; b[N5+p]=1-B
    for p in range(81):
        W[OWN+p,G1]=1; W[OPP+p,G1]=1; W[OWN+p,G2]=1; W[OPP+p,G2]=1
    b[G1]=-Q['s0']; b[G2]=-Q['s1']
    T['L6.W']=W; T['L6.b']=b
    W=np.zeros((512,82),np.float32); b=np.zeros(82,np.float32)
    for e in range(81):
        W[LIN+e,e]=1; W[SA+e,e]=-Q['w_sa']; W[NE+e,e]=-Q['w_ne']; W[INFP+e,e]=-Q['alpha']
        W[SAC+e,e]=-Q['w_conn']
        for n in NB[e]: W[N5+n,e]+=Q['W_cap']
        b[e]=Q['prior'][line(e)]-Q['C']
        dl=(Q['prior_late'][line(e)]-Q['prior'][line(e)])/(Q['s1']-Q['s0'])
        W[G1,e]+=dl; W[G2,e]-=dl
    b[81]=Q['pass_logit']
    T['out.W']=W; T['out.b']=b
    return T
