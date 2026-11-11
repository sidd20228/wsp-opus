Results on the test positions:
- The game-69 and game-51/92 blunders are now avoided: the net picks a different move or passes.
- Positions 6, 7 and 8 still show high-probability fake captures. These come from a long chain (witness more than 4 hops away) or a parity collision, and the capture signal exempts the self-atari penalty.

I'll measure how often true versus fake capture signals occur on such self-atari-shaped moves.

```python
import json, numpy as np
from collections import Counter, deque
d='/work/pass_2/'
G=json.load(open(d+'games.json')); B=np.load(d+'boards.npy'); V=np.load(d+'valid.npy')
def nbrs(p):
    r,c=divmod(p,9); o=[]
    for dr,dc in((-1,0),(1,0),(0,-1),(0,1)):
        rr,cc=r+dr,c+dc
        if 0<=rr<9 and 0<=cc<9:o.append(rr*9+cc)
    return o
NB=[nbrs(p) for p in range(81)]
PAR=[(p//9+p%9)%2 for p in range(81)]
def label(b):
    lab=-np.ones(81,int); ch=[]
    for p in range(81):
        if b[p]!=0 and lab[p]<0:
            col=b[p]; st=[p]; lab[p]=len(ch); mem=[p]
            while st:
                x=st.pop()
                for q in NB[x]:
                    if b[q]==col and lab[q]<0: lab[q]=len(ch); st.append(q); mem.append(q)
            libs=set(q for x in mem for q in NB[x] if b[q]==0)
            ch.append((col,mem,libs))
    return lab,ch
def witness(b,lab,ch,n,e,maxh=4):
    col,mem,libs=ch[lab[n]]
    dist={n:0}; dq=deque([n])
    while dq:
        x=dq.popleft()
        nl=[q for q in NB[x] if b[q]==0]
        if len(nl)>=2 or any(PAR[q]!=PAR[e] for q in nl): return dist[x]
        for q in NB[x]:
            if lab[q]==lab[n] and q not in dist: dist[q]=dist[x]+1; dq.append(q)
    return 99
stat=Counter()
for gi,g in enumerate(G):
    me=1 if g['net_color']=='black' else -1
    for m in g['moves']:
        if m['by']!='net': continue
        k=m['ply']-1
        b=B[gi,k] if k>=0 else np.zeros(81,int)
        lab,ch=label(b)
        for e in range(81):
            if b[e]!=0: continue
            claim=False; true=False
            for n in NB[e]:
                if b[n]==-me:
                    libs=ch[lab[n]][2]
                    if len(libs)==1: true=True; claim=True
                    elif witness(b,lab,ch,n,e)>4: claim=True
            if not claim: continue
            # self-atari shape ignoring capture: resulting libs
            E=[q for q in NB[e] if b[q]==0]
            newlibs=set(E)
            for n in NB[e]:
                if b[n]==me: newlibs|=ch[lab[n]][2]
            newlibs.discard(e)
            shape = len(newlibs)<=1
            stat[('true' if true else 'fake', 'SAshape' if shape else 'ok')]+=1
print(stat)
```