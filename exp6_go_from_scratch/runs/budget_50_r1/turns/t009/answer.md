State doesn't persist between runs; I'll put analysis helpers in a file.

```python
ana_src = r'''
import numpy as np, json
from collections import deque
exec(open('/work/go.py').read())
def load_pass(k):
    import os
    dd='/work/pass_%d/'%k
    r={}
    for f in os.listdir(dd):
        if f.endswith('.npy'): r[f[:-4]]=np.load(dd+f, allow_pickle=True)
    return r
def boards_from_tests(d):
    boards=[]
    for name,b,tm in d['T']: boards.append((parse(b), 1 if tm=='black' else 2))
    for name,m in d['R']:
        g=Game()
        for x in m: g.play(x)
        boards.append((list(g.b), g.col))
    return boards
def e(b,q): return sum(1 for y in NB[q] if b[y]==0)
def hv_witness(b,v):
    c=b[v]; r,cc=divmod(v,9); best=0
    for seg in ([r*9+x for x in (cc-1,cc,cc+1) if 0<=x<9],[y*9+cc for y in (r-1,r,r+1) if 0<=y<9]):
        best=max(best,sum(e(b,q) for q in seg if b[q]==c))
    return best>=2
def chain_dist(b,src):
    c=b[src]; dist={src:0}; dq=deque([src])
    while dq:
        x=dq.popleft()
        for y in NB[x]:
            if b[y]==c and y not in dist: dist[y]=dist[x]+1; dq.append(y)
    return dist
'''
open('/work/ana.py','w').write(ana_src)
exec(open('/work/ana.py').read())
d=json.load(open('/work/tests1.json')); lay=json.load(open('/work/layout_v1.json'))
P=load_pass(1); boards=boards_from_tests(d); L6=P['act_L6']
done=set()
for i,(b,col) in enumerate(boards):
    for p in range(81):
        if b[p]==0: continue
        nm='AtO' if b[p]!=col else 'AtM'
        g_,l=group(b,p)
        if len(l)>1 and L6[i,lay[5][nm]+p]>0.5:
            key=(i,min(g_))
            dist=chain_dist(b,p); ws=[q for q in dist if hv_witness(b,q)]
            print(nm,i,p,'size',len(g_),'libs',len(l),'nwit',len(ws),'mindist',min([dist[q] for q in ws]) if ws else None)
```