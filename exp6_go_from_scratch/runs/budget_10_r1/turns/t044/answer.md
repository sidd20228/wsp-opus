```python
import json, numpy as np
B=np.load('/work/pass_5/boards.npy'); Vd=np.load('/work/pass_5/valid.npy')
G=json.load(open('/work/pass_5/games.json'))
rng=np.random.default_rng(7)
lst=[]; meta=[]
for gi in rng.permutation(len(G))[:100]:
    n=int(Vd[gi].sum())
    ply=int(rng.integers(20,n-1))
    bb=B[gi][ply]
    tm='white' if ply%2==0 else 'black'   # board after ply; next mover
    lst.append({"board":["".join({0:'.',1:'X',-1:'O'}[int(v)] for v in bb[r*9:r*9+9]) for r in range(9)],"to_move":tm})
    meta.append((int(gi),ply))
json.dump({'meta':meta,'req':lst},open('/work/pos6_sel.json','w'))
print(json.dumps({"positions":lst,"activations":True},separators=(',',':')))
```