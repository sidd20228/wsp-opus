Parity is actually a good code: only 497 of about 50,000 two-liberty chains collide (1%), and mod-3 codes are much worse. The false captures must therefore come from limited propagation reach on long chains. Let me measure how many chain hops separate the stones next to a chain's different liberties.

```python
import json, numpy as np
from collections import Counter, deque
d='/work/pass_2/'
B=np.load(d+'boards.npy'); V=np.load(d+'valid.npy')
def nbrs(p):
    r,c=divmod(p,9); o=[]
    for dr,dc in((-1,0),(1,0),(0,-1),(0,1)):
        rr,cc=r+dr,c+dc
        if 0<=rr<9 and 0<=cc<9:o.append(rr*9+cc)
    return o
NB=[nbrs(p) for p in range(81)]
PAR=[(p//9+p%9)%2 for p in range(81)]
def chains(b):
    seen=-np.ones(81,int); out=[]
    for p in range(81):
        if b[p]!=0 and seen[p]<0:
            col=b[p]; st=[p]; seen[p]=len(out); mem=[p]
            while st:
                x=st.pop()
                for q in NB[x]:
                    if b[q]==col and seen[q]<0: seen[q]=len(out); st.append(q); mem.append(q)
            libs=set(q for x in mem for q in NB[x] if b[q]==0)
            out.append((col,mem,libs))
    return out
# For each (chain with >=2 libs, lib e, stone n adjacent to e): hop distance from n to nearest "witness" stone
# (witness = stone with >=2 adjacent libs, or stone adjacent to a lib of other parity than e)
H=Counter(); total=0
for gi in range(B.shape[0]):
    for k in range(162):
        if not V[gi,k]: continue
        b=B[gi,k]
        for col,mem,libs in chains(b):
            if len(libs)<2: continue
            ms=set(mem)
            nl={x:[q for q in NB[x] if b[q]==0] for x in mem}
            for e in libs:
                for n in NB[e]:
                    if n not in ms: continue
                    total+=1
                    # BFS
                    dist={n:0}; dq=deque([n]); found=None
                    while dq:
                        x=dq.popleft()
                        if len(nl[x])>=2 or any(PAR[q]!=PAR[e] for q in nl[x]): found=dist[x]; break
                        for q in NB[x]:
                            if q in ms and q not in dist: dist[q]=dist[x]+1; dq.append(q)
                    H[found if found is not None else 'never']+=1
print(total)
for k in sorted(H,key=lambda x:(isinstance(x,str),x)): print(k,H[k],round(H[k]/total*100,3))
```