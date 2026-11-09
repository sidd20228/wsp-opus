```python
import json, numpy as np
exec(open('/work/go.py').read())
B=np.load('/work/pass_4/boards.npy'); Vd=np.load('/work/pass_4/valid.npy')
tot=0; mixed=0; npos=0
for gi in range(B.shape[0]):
    n=int(Vd[gi].sum())
    for ply in range(0,n,2):
        b=np.where(B[gi][ply]==-1,2,B[gi][ply]).astype(int)
        for col in (1,2):
            for p in range(81):
                if b[p]!=0: continue
                oc=[y for y in NB[p] if b[y]==col]
                if not oc: continue
                at=[chain(b,y)[1]=={p} for y in oc]
                if any(at):
                    tot+=1
                    if not all(at): mixed+=1
print('capture points',tot,'mixed',mixed)
```