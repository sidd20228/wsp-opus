No missed ataris (as designed), but some false ataris on big groups. Let me diagnose whether they lack any witness or the witness is just beyond flood radius.

```python
from collections import deque
def hv_witness(b,v):
    c=b[v]
    def e(q): return sum(1 for y in NB[q] if b[y]==0)
    r,cc=divmod(v,9)
    best=0
    for seg in ([r*9+x for x in (cc-1,cc,cc+1) if 0<=x<9],[y*9+cc for y in (r-1,r,r+1) if 0<=y<9]):
        s=sum(e(q) for q in seg if b[q]==c)
        best=max(best,s)
    return best>=2
def chain_dist(b,src):
    c=b[src]; dist={src:0}; dq=deque([src])
    while dq:
        x=dq.popleft()
        for y in NB[x]:
            if b[y]==c and y not in dist: dist[y]=dist[x]+1; dq.append(y)
    return dist
seen=set(); rows=[]
for (i,p,sz,nl) in fa_examples:
    b,col=boards[i]
    dist=chain_dist(b,p)
    ws=[q for q in dist if hv_witness(b,q)]
    md=min([dist[q] for q in ws]) if ws else None
    rows.append((i,p,sz,nl,len(ws),md))
for r in rows: print(r)
# also own false
for i,(b,col) in enumerate(boards):
    for p in range(81):
        if b[p]==col:
            g_,l=group(b,p)
            if len(l)>1 and acts[5][i,lay[5]['AtM']+p]>0.5:
                dist=chain_dist(b,p); ws=[q for q in dist if hv_witness(b,q)]
                print('own',i,p,len(g_),len(l),len(ws),min([dist[q] for q in ws]) if ws else None)
```