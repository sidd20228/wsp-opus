The remaining fakes come from parity collisions on big chains, or witnesses 7–13 hops away on 12–28-stone walls. Let me see whether the adjacent stone's own connectivity (how many same-colour neighbours it has) separates true captures from fakes, using all claims from passes 2 and 5.

```python
import json, numpy as np
from collections import Counter, deque
exec(open('/work/build5.py').read().split("# L1-L4 layout")[0])
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
stat=Counter(); sizes=Counter()
for d in ['/work/pass_2/','/work/pass_5/']:
    G=json.load(open(d+'games.json')); B=np.load(d+'boards.npy')
    for gi,g in enumerate(G):
        me=1 if g['net_color']=='black' else -1
        for m in g['moves']:
            if m['by']!='net' or m['ply']==0: continue
            b=B[gi,m['ply']-1]; lab,ch=label(b)
            for e in range(81):
                if b[e]!=0: continue
                for n in NB[e]:
                    if b[n]!=-me: continue
                    libs=ch[lab[n]][2]
                    if len(libs)==1: kind='true'
                    elif witness(b,lab,n,e)>5: kind='fake'
                    else: continue
                    k=sum(b[q]==-me for q in NB[n])
                    stat[(kind,k)]+=1
                    sizes[(kind,min(8,len(ch[lab[n]][1])))]+=1
for k in range(5): print(k,'true',stat[('true',k)],'fake',stat[('fake',k)])
for s in range(1,9): print('size',s,'true',sizes[('true',s)],'fake',sizes[('fake',s)])
```