Local features don't separate true from fake captures. Instead I'll restructure L6:
- **Linear terms:** escape and territory influence become linear terms in LIN, which frees two units per point.
- **Fifth hop:** the freed units carry a fifth propagation hop for opponent atari flags. The capture bonus now comes from 5-hop information.
- **Self-atari exemption:** a capture claim only half-exempts a move from the self-atari penalty, so far-reaching fakes are no longer attractive.

```python
src=open('/work/build3.py').read()
start=src.index('    # ---------- L6')
end=src.index("    T['L6.W']=W")
new_l6 = r'''    # ---------- L6
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
    # 5th hop for opponent flags (negated): N5_a(p)=relu(1 - sum_{q in N[p]} (OPP(q)-NX_a(q)) - B + B*OPP(p))
    for p in range(81):
        for src_,dst in ((NX0,N50),(NX1,N51)):
            for q in NB[p]+[p]:
                W[OPP+q,dst+p]-=1; W[src_+q,dst+p]+=1
            W[OPP+p,dst+p]+=B; b[dst+p]=1-B
'''
src=src[:start]+new_l6+src[end:]
src=src.replace("LIN,ESC,SA,NE,INFP,INFN=0,81,162,243,324,405","LIN,SA,NE,INFP,N50,N51=0,81,162,243,324,405")
# output layer
ostart=src.index("    W=np.zeros((512,82),np.float32)")
oend=src.index("    T['out.W']=W")
new_out=r'''    W=np.zeros((512,82),np.float32); b=np.zeros(82,np.float32)
    for e in range(81):
        W[LIN+e,e]=1; W[SA+e,e]=-Q['w_sa']; W[NE+e,e]=-Q['w_ne']; W[INFP+e,e]=-Q['alpha']
        for n in NB[e]:
            W[N50+n,e]+=Q['W_cap']; W[N51+n,e]+=Q['W_cap']
        b[e]=Q['prior'][line(e)]-Q['C']
    b[81]=Q['pass_logit']
'''
src=src[:ostart]+new_out+src[oend:]
src=src.replace("w_esc=2.5,","W_esc=6.0,").replace("lin_inf=0.0","lin_inf=0.3").replace("ax=4.0","ax=2.0")
open('/work/build4.py','w').write(src)
exec(src)
print(DEFAULT)
T=build()
# sanity: LIN lower bound >= 0 ? C + W_E*E + W_c*s_x + W_esc*a_o + lin_inf*inf ; inf min ~ -11*0.3=-3.3
for k,v in T.items(): save(k,v)
print('saved v4')
```