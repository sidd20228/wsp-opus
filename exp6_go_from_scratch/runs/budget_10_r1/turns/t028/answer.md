```python
import json, numpy as np
G=json.load(open('/work/pass_4/games.json'))
B=np.load('/work/pass_4/boards.npy'); Vd=np.load('/work/pass_4/valid.npy')
for gi in [34,35]+[i for i,g in enumerate(G) if g['end']=='repetition'][2:8]:
    n=int(Vd[gi].sum()); last=B[gi][n-1]
    same=[ply for ply in range(n-1) if (B[gi][ply]==last).all()]
    diff=[(ply,int((B[gi][ply]!=last).sum())) for ply in range(max(0,n-10),n-1)]
    print(gi,n,'same as',same,'diffs',diff)
```