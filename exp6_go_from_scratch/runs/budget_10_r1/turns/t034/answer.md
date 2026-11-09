```python
import json, numpy as np
exec(open('/work/go.py').read())
G=json.load(open('/work/pass_4/games.json'))
B=np.load('/work/pass_4/boards.npy')
tot=0; topcap=0; ex=[]
for gi,g in enumerate(G):
    me=1 if g['net_color']=='black' else 2; op=3-me
    for m in g['moves']:
        if m['by']!='net': continue
        ply=m['ply']
        b=np.where(B[gi][ply-1]==-1,2,B[gi][ply-1]).astype(int) if ply>0 else np.zeros(81,int)
        caps=set()
        for p in range(81):
            if b[p]!=0: continue
            if any(b[y]==op and chain(b,y)[1]=={p} for y in NB[p]): caps.add(p)
        if caps:
            tot+=1
            t=m['top5'][0][0]
            if t in caps: topcap+=1
            elif len(ex)<5: ex.append((gi,ply,t,sorted(caps),m['top5'][:3]))
print(tot,topcap)
for e in ex: print(e)
```