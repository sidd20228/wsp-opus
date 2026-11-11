The dominant pattern now is a double ko, where my net passes with probability 1.0 at the critical moment:
1. I take ko A.
2. The opponent takes ko B.
3. I pass.
4. The opponent retakes A.
5. I retake B, which repeats a position and loses.

Playing any harmless stone instead of passing would break the cycle, so the pass logit must sit below cheap tempo moves. There are also a few remaining three-move cycles from fake captures; let me check their witness distances.

```python
import json, numpy as np
from collections import Counter, deque
exec(open('/work/build5.py').read().split("# L1-L4 layout")[0])
d='/work/pass_5/'
G=json.load(open(d+'games.json')); B=np.load(d+'boards.npy')
def chain_of(b,p):
    col=b[p]; mem=[p]; s={p}; st=[p]
    while st:
        x=st.pop()
        for q in NB[x]:
            if b[q]==col and q not in s: s.add(q); st.append(q); mem.append(q)
    libs=set(q for x in mem for q in NB[x] if b[q]==0)
    return mem,libs
def witness(b,n,e):
    mem,libs=chain_of(b,n); ms=set(mem)
    dist={n:0}; dq=deque([n])
    while dq:
        x=dq.popleft()
        nl=[q for q in NB[x] if b[q]==0]
        if len(nl)>=2 or any(PAR[q]!=PAR[e] for q in nl): return dist[x]
        for q in NB[x]:
            if q in ms and q not in dist: dist[q]=dist[x]+1; dq.append(q)
    return None
for gi,k in [(0,None),(55,None),(78,None),(85,None),(107,None),(113,None),(117,None)]:
    g=G[gi]; mv=g['moves']; n=len(mv); last=B[gi,n-1]
    k=[j for j in range(n-1) if (B[gi,j]==last).all()][-1]
    e=mv[k+1]['move']; b=B[gi,k]; me=1 if g['net_color']=='black' else -1
    out=[]
    for q in NB[e]:
        if b[q]==-me:
            mem,l=chain_of(b,q); out.append((len(mem),len(l),witness(b,q,e),sorted(divmod(x,9) for x in l)))
    print(gi,divmod(e,9),out, 'top5',mv[k+1]['top5'][:3])
```