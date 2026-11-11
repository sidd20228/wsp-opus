The cause is visible on the board. During the middle and late game my net fills its own territory with checkerboard patterns ("X.X.X.X.X" along the edge). That leaves many disconnected single stones with false eyes, which the opponent then captures one by one while the net passes. Filling own territory gains nothing under area scoring. Let me measure how often my moves land inside already-secure own territory, and how strongly the influence penalty actually applies there.

```python
import json, numpy as np
from collections import Counter
exec(open('/work/build14.py').read().split("CC,F0O")[0])
d='/work/pass_14/'
G=json.load(open(d+'games.json')); B=np.load(d+'boards.npy')
iw={1:1.0,2:0.5,3:0.25}
def region_owner(b,e):
    st=[e]; seen={e}; bd=set(); size=1
    while st:
        x=st.pop()
        for q in NB[x]:
            if b[q]==0 and q not in seen: seen.add(q); st.append(q)
            elif b[q]!=0: bd.add(int(b[q]))
    return bd,len(seen)
cnt=Counter(); infs=[]; sizes=Counter()
for gi,g in enumerate(G):
    me=1 if g['net_color']=='black' else -1
    for m in g['moves']:
        if m['by']!='net' or m['move']==81: continue
        k=m['ply']; e=m['move']; b=B[gi,k-1]
        bd,sz=region_owner(b,e)
        own_terr = bd=={me}
        S=int((b!=0).sum())
        cnt[(S>=40, own_terr)]+=1
        if own_terr:
            inf=sum(iw[mdist(e,q)]*b[q]*me for q in range(81) if mdist(e,q) in iw)
            infs.append(inf); sizes[min(sz,10)]+=1
print(cnt)
print('inf percentiles for own-territory fills',np.percentile(infs,[10,25,50,75,90]))
print('region sizes',sorted(sizes.items()))
print('per game own-terr fills', sum(v for (a,b),v in cnt.items() if b)/len(G))
```