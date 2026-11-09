```python
exec(open('/work/an.py').read())
from collections import deque, Counter
def reach_libs(b,s,R):
    col=b[s]; dist={s:0}; dq=deque([s]); libs=set()
    while dq:
        x=dq.popleft()
        for y in NB[x]:
            if b[y]==0: libs.add(y)
            elif b[y]==col and y not in dist and dist[x]<R:
                dist[y]=dist[x]+1; dq.append(y)
    return libs
A,X,lg=load_pass(6)
EDGES=[(a,b) for a in range(81) for b in NB[a] if a<b]
st=Counter(); bad=[]
for i in range(len(X)):
    b,me=board_from_input(X[i]); op=3-me
    for e,(a,bb) in enumerate(EDGES):
        val=A[6][i][e]
        exp=0
        for s,t in ((a,bb),(bb,a)):
            if b[s]==op and b[t]==0 and reach_libs(b,s,4)=={t}: exp=1
        both_opp = b[a]==op and b[bb]==op
        if both_opp: continue
        st[(exp, round(float(val),3))]+=1
        if abs(val-exp)>1e-3 and len(bad)<5: bad.append((i,a,bb,val,exp))
    # SA, Z1, EYE
    for p in range(81):
        if b[p]!=0: continue
        E=sum(b[y]==0 for y in NB[p]); no=sum(b[y]==me for y in NB[p]); nop=sum(b[y]==op for y in NB[p]); d=len(NB[p])
        sV=E+2*nop
        sa=max(0,2-E+2*(sV-2*d+1)); z1=max(0,no-3*E+3); eye=max(0,no-d+1)
        for k,ref,u in (('sa',sa,144+p),('z1',z1,225+p),('eye',eye,306+p)):
            if abs(A[6][i][u]-ref)>1e-3: st['bad_'+k]+=1
print(st); print(bad)
```