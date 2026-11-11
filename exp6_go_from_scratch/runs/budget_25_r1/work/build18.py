
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
OWN,OPP,AO,PM,NG,NXP=0,81,162,243,324,405           # L5
LIN,SA,NE,INFP,KO,Z=0,81,162,243,324,405            # L6
G1,G2=486,487
DEFAULT=dict(K=5.0,B=1e4,C=10.0,W_cap=12.0,W_c=1.5,W_o=0.3,W_E=0.5,W_esc=6.0,W_def=1.0,
             w_sa=6.0,w_ne=15.0,ne_a=3.0,ne_b=4.0,ne_c=4.0,ne_d=1.0,ko_ao=1.0,w_ko=30.0,w_z=8.0,alpha=3.0,t1=2.5,lin_inf=0.3,
             infw={1:1.0,2:0.5,3:0.25},
             prior={0:-2.0,1:1.0,2:1.6,3:0.8,4:1.0},
             prior_late={0:-1.0,1:0.3,2:0.6,3:0.4,4:0.4},s0=40.0,s1=70.0,
             pass_logit=-6.0,
             sa=dict(k0=4.0,E=2.0,t=4.0,w=1.0,ax=2.0),
             z=dict(wl=2.0,E=4.0,safe=3.0))
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
        nterm((F0O if (1-pi)==0 else F1O)+p,0,0)
        nterm(GO+p,1,0)
        nterm((F0X if (1-pi)==0 else F1X)+p,0,1)
        nterm((F0X if pi==0 else F1X)+p,1,1)
    T['L1.W']=W; T['L1.b']=b
    # ---------- L2,L3 propagate; L4 own negated (hop 3), opp propagate (hop 3)
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
    # ---------- L5
    W=np.zeros((512,512),np.float32); b=np.zeros(512,np.float32)
    for p in range(81):
        W[CC+p,OWN+p]=1; b[OWN+p]=-1
        W[CC+p,OPP+p]=-1; b[OPP+p]=1
        for blk in (F0O,F1O,GO): W[blk+p,AO+p]=1
        b[AO+p]=-1
        W[F0O+p,PM+p]=1; W[F1O+p,PM+p]=1
        W[GO+p,NG+p]=1
        srcb=F0X if PAR[p]==0 else F1X
        for q in NB[p]+[p]: W[srcb+q,NXP+p]=-1
        W[CC+p,NXP+p]=-B; b[NXP+p]=1
    T['L5.W']=W; T['L5.b']=b
    # ---------- L6
    # own classes for neighbour n: AO(n); WL=PM-AO; WG=NG-AO; S3=OWN-PM-NG+AO
    W=np.zeros((512,512),np.float32); b=np.zeros(512,np.float32)
    iw=Q['infw']; sa=Q['sa']; zz=Q['z']
    def add_S3(u,n,coef):
        W[OWN+n,u]+=coef; W[PM+n,u]-=coef; W[NG+n,u]-=coef; W[AO+n,u]+=coef
    for e in range(81):
        deg=len(NB[e])
        # LIN
        u=LIN+e; b[u]=Q['C']+Q['W_E']*deg
        for n in NB[e]:
            W[OPP+n,u]+=Q['W_c']-Q['W_E']; W[OWN+n,u]+=Q['W_o']-Q['W_E']
            W[AO+n,u]+=Q['W_esc']
            # W_def * w, w = OWN - AO - S3
            W[OWN+n,u]+=Q['W_def']; W[AO+n,u]-=Q['W_def']; add_S3(u,n,-Q['W_def'])
            W[NXP+n,u]+=Q['W_cap']
        # SA = relu(k0 - cE*E - ct*t - cw*w - cax*a_x)
        u=SA+e; k0,cE,ct,cw,cax=sa['k0'],sa['E'],sa['t'],sa['w'],sa['ax']
        b[u]=k0-cE*deg
        for n in NB[e]:
            W[OWN+n,u]+=cE; W[OPP+n,u]+=cE
            # -ct*t - cw*w = -ct*S3 - cw*(OWN-AO-S3) = (cw-ct)*S3 - cw*OWN + cw*AO
            add_S3(u,n,cw-ct); W[OWN+n,u]-=cw; W[AO+n,u]+=cw
            W[NXP+n,u]-=cax
        # NE = relu(na*(s_o - 2 s_x - deg) + nb - nc*a_o + nd*w)
        u=NE+e; b[u]=-Q['ne_a']*deg+Q['ne_b']
        for n in NB[e]:
            W[OWN+n,u]+=Q['ne_a']; W[OPP+n,u]-=2*Q['ne_a']; W[AO+n,u]-=Q['ne_c']
            W[OWN+n,u]+=Q['ne_d']; W[AO+n,u]-=Q['ne_d']; add_S3(u,n,-Q['ne_d'])
        # KO = relu(s_x - deg + 2 - a_x)
        u=KO+e; b[u]=2-deg
        for n in NB[e]: W[OPP+n,u]+=1; W[NXP+n,u]-=1; W[AO+n,u]+=Q['ko_ao']
        # Z = relu(zwl*wl - zE*E - zs*(S3+WG)),  wl = PM-AO, S3+WG = OWN-PM
        u=Z+e; b[u]=-zz['E']*deg
        for n in NB[e]:
            W[OWN+n,u]+=zz['E']; W[OPP+n,u]+=zz['E']
            W[PM+n,u]+=zz['wl']; W[AO+n,u]-=zz['wl']
            W[OWN+n,u]-=zz['safe']; W[PM+n,u]+=zz['safe']
        b[INFP+e]=-Q['t1']
        for q in range(81):
            d=mdist(e,q)
            if d in iw:
                W[OWN+q,INFP+e]+=iw[d]; W[OPP+q,INFP+e]-=iw[d]
                W[OWN+q,LIN+e]+=Q['lin_inf']*iw[d]; W[OPP+q,LIN+e]-=Q['lin_inf']*iw[d]
    for p in range(81):
        W[OWN+p,G1]=1; W[OPP+p,G1]=1; W[OWN+p,G2]=1; W[OPP+p,G2]=1
    b[G1]=-Q['s0']; b[G2]=-Q['s1']
    T['L6.W']=W; T['L6.b']=b
    W=np.zeros((512,82),np.float32); b=np.zeros(82,np.float32)
    for e in range(81):
        W[LIN+e,e]=1; W[SA+e,e]=-Q['w_sa']; W[NE+e,e]=-Q['w_ne']; W[INFP+e,e]=-Q['alpha']
        W[KO+e,e]=-Q['w_ko']; W[Z+e,e]=-Q['w_z']
        b[e]=Q['prior'][line(e)]-Q['C']
        dl=(Q['prior_late'][line(e)]-Q['prior'][line(e)])/(Q['s1']-Q['s0'])
        W[G1,e]+=dl; W[G2,e]-=dl
    b[81]=Q['pass_logit']
    T['out.W']=W; T['out.b']=b
    return T
