A key observation: a fake capture with no own neighbour and no empty neighbour is suicide, so it's illegal and harmless. The dangerous fakes are those that connect to my stones (211 cases), against 173 true captures of that shape. Let me see whether the local stone balance at distance 2 separates them.

```python
import json, numpy as np
from collections import Counter, deque
exec(open('/work/build3.py').read().split("# L1-L4 layout")[0])
d='/work/pass_2/'
G=json.load(open(d+'games.json')); B=np.load(d+'boards.npy')
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
            claim=False; true=False
            for n in NB[e]:
                if b[n]==-me:
                    libs=ch[lab[n]][2]
                    if len(libs)==1: true=True; claim=True
                    elif witness(b,lab,n,e)>4: claim=True
            if not claim: continue
            E=[q for q in NB[e] if b[q]==0]
            newlibs=set(E)
            for n in NB[e]:
                if b[n]==me: newlibs|=ch[lab[n]][2]
            newlibs.discard(e)
            if len(newlibs)>1: continue
            so=sum(b[n]==me for n in NB[e]); sx=sum(b[n]==-me for n in NB[e])
            if so==0: continue
            w={1:1.0,2:0.5,3:0.25}
            inf=sum(w[md(q,e)]*b[q]*me for q in range(81) if md(q,e) in w)
            x2=sum(b[q]==-me for q in range(81) if md(q,e)==2)
            o2=sum(b[q]==me for q in range(81) if md(q,e)==2)
            rows.append((true,inf,x2-o2,sx,len(E)))
rows=np.array(rows)
for t in (1,0):
    R=rows[rows[:,0]==t]; print('true' if t else 'fake',len(R))
    print(' inf',np.percentile(R[:,1],[10,25,50,75,90]))
    print(' x2-o2',Counter(R[:,2]).most_common())
    print(' sx',Counter(R[:,3]).most_common())
```