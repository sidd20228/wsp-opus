```python
exec(open('/work/an.py').read())
from collections import deque
def reach_libs(b,s,R):
    col=b[s]; dist={s:0}; dq=deque([s]); libs=set()
    while dq:
        x=dq.popleft()
        for y in NB[x]:
            if b[y]==0: libs.add(y)
            elif b[y]==col and y not in dist and dist[x]<R:
                dist[y]=dist[x]+1; dq.append(y)
    return libs
SHAPES_=SHAPES
exec(open('/work/build2.py').read())
P=PV2; K=P['K']
A,X,lg=load_pass(6)
mx=0
for i in range(len(X)):
    b,me=board_from_input(X[i]); op=3-me
    own=(b==me).astype(float); opp=(b==op).astype(float)
    infl=np.zeros(81)
    for p in range(81):
        r,c=divmod(p,9)
        for q in range(81):
            r2,c2=divmod(q,9); d2=(r-r2)**2+(c-c2)**2
            if d2 and d2 in K: infl[p]+=K[d2]*(own[q]-opp[q])
    emp=(b==0); ta=np.where(emp,np.maximum(0,P['T']-infl),0); tb=np.where(emp,np.maximum(0,-infl),0)
    for p in range(81):
        if b[p]!=0: continue
        r,c=divmod(p,9)
        gain=sum(P['K2'][(r-q//9)**2+(c-q%9)**2]*P['alpha']*(ta[q]-P['gamma']*tb[q]) for q in range(81) if (r-q//9)**2+(c-q%9)**2 in P['K2'])
        E=sum(b[y]==0 for y in NB[p]); no=sum(b[y]==me for y in NB[p]); nop=sum(b[y]==op for y in NB[p]); d=len(NB[p])
        sa=max(0,2-E+2*(E+2*nop-2*d+1)); z1=max(0,no-3*E+3); eye=max(0,no-d+1)
        ncap=sum(1 for s in NB[p] if b[s]==op and reach_libs(b,s,4)=={p})
        line=min(r,c,8-r,8-c)
        exp=P['bias']+P['linebias'][line]+gain+P['wcap']*ncap-P['wsa']*sa-P['wz1']*z1-P['weye']*eye
        mx=max(mx,abs(exp-lg[i][p]))
print('max logit err',mx, 'pass logit',lg[0][81])
```