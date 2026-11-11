Fakes are common: 632 fake versus 1,398 true capture signals on self-atari-shaped moves. Let me look for local features that separate the two groups.

```python
import json, numpy as np
from collections import Counter, deque
d='/work/pass_2/'
G=json.load(open(d+'games.json')); B=np.load(d+'boards.npy')
def nbrs(p):
    r,c=divmod(p,9); o=[]
    for dr,dc in((-1,0),(1,0),(0,-1),(0,1)):
        rr,cc=r+dr,c+dc
        if 0<=rr<9 and 0<=cc<9:o.append(rr*9+cc)
    return o
NB=[nbrs(p) for p in range(81)]
PAR=[(p//9+p%9)%2 for p in range(81)]
def md(p,q): return abs(p//9-q//9)+abs(p%9-q%9)
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
def witness(b,lab,n,e):
    dist={n:0}; dq=deque([n])
    while dq:
        x=dq.popleft()
        nl=[q for q in NB[x] if b[q]==0]
        if len(nl)>=2 or any(PAR[q]!=PAR[e] for q in nl): return dist[x]
        for q in NB[x]:
            if lab[q]==lab[n] and q not in dist: dist[q]=dist[x]+1; dq.append(q)
    return 99
rows=[]
for gi,g in enumerate(G):
    me=1 if g['net_color']=='black' else -1
    for m in g['moves']:
        if m['by']!='net': continue
        k=m['ply']-1
        b=B[gi,k] if k>=0 else np.zeros(81,int)
        lab,ch=label(b)
        for e in range(81):
            if b[e]!=0: continue
            claim=False; true=False; sizes=[]
            for n in NB[e]:
                if b[n]==-me:
                    libs=ch[lab[n]][2]
                    if len(libs)==1: true=True; claim=True; sizes.append(len(ch[lab[n]][1]))
                    elif witness(b,lab,n,e)>4: claim=True; sizes.append(len(ch[lab[n]][1]))
            if not claim: continue
            E=[q for q in NB[e] if b[q]==0]
            newlibs=set(E)
            for n in NB[e]:
                if b[n]==me: newlibs|=ch[lab[n]][2]
            newlibs.discard(e)
            if len(newlibs)>1: continue
            sx=sum(b[n]==-me for n in NB[e]); so=sum(b[n]==me for n in NB[e])
            x2=sum(b[q]==-me for q in range(81) if md(q,e)==2)
            o2=sum(b[q]==me for q in range(81) if md(q,e)==2)
            rows.append((true,max(sizes),len(E),so,sx,x2,o2))
rows=np.array(rows)
for t in (1,0):
    R=rows[rows[:,0]==t]
    print('true' if t else 'fake', len(R))
    print(' size',Counter(np.minimum(R[:,1],8)).most_common())
    print(' E',Counter(R[:,2]).most_common(),' so',Counter(R[:,3]).most_common(),' sx',Counter(R[:,4]).most_common())
    print(' x2',Counter(R[:,5]).most_common(),' o2',Counter(R[:,6]).most_common())
```