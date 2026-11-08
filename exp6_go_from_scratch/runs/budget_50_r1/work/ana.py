
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
