```python
import json, numpy as np
exec(open('/work/go.py').read())
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
def classify(b,p,me):
    op=3-me
    oc=[y for y in NB[p] if b[y]==op]
    real=any(chain(b,y)[1]=={p} for y in oc)
    fake=(not real) and any(reach_libs(b,y,3)=={p} for y in oc)
    b2=b.copy(); b2[p]=me; ncap=0
    for y in NB[p]:
        if b2[y]==op and not chain(b2,y)[1]:
            ch=chain(b2,y)[0]; ncap+=len(ch)
            for z in ch: b2[z]=0
    al=len(chain(b2,p)[1]); E=sum(b[y]==0 for y in NB[p]); no=sum(b[y]==me for y in NB[p])
    return ('cap%d'%ncap if real else ('FAKE' if fake else 'nc'), 'E%d'%E, 'own%d'%no,'al%d'%min(al,2))
res=Counter()
G=json.load(open(f'/work/pass_5/games.json'))
B=np.load(f'/work/pass_5/boards.npy'); Vd=np.load(f'/work/pass_5/valid.npy')
for gi,g in enumerate(G):
    if g['end']!='repetition': continue
    n=int(Vd[gi].sum()); last=B[gi][n-1]
    j=max(ply for ply in range(n-1) if (B[gi][ply]==last).all())
    me=1 if g['net_color']=='black' else 2
    seq=[]
    for ply in range(j+1,n):
        m=g['moves'][ply]
        if m['move']==81: seq.append((m['by'],'pass')); continue
        b=np.where(B[gi][ply-1]==-1,2,B[gi][ply-1]).astype(int)
        mover=me if m['by']=='net' else 3-me
        seq.append((m['by'][0],)+classify(b,m['move'],mover))
    res[tuple(seq)]+=1
for k,v in res.most_common(20): print(v,k)
```