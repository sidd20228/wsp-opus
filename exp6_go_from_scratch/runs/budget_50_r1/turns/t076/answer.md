Before testing, a concern: this could also block *real* captures at ε=0 points that touch my own weak stones (common in capturing races). Let me count real vs. false captures at such points in recent games.

```python
exec(open('/work/ana.py').read())
from collections import Counter
cnt=Counter()
for pk in [13,14]:
    G=json.load(open('/work/pass_%d/games.json'%pk)); Bd=np.load('/work/pass_%d/boards.npy'%pk)
    def tob(arr): return [1 if x==1 else (2 if x==-1 else 0) for x in arr]
    for gi,g in enumerate(G):
        for k in range(10,len(g['moves'])-1):
            nxt=g['moves'][k+1]
            if nxt['move']==81: continue
            b=tob(Bd[gi,k]); col=1 if (k+1)%2==0 else 2   # ply k+1 even -> black to move
            for p in range(81):
                if b[p] or any(b[q]==0 for q in NB[p]): continue
                own=[q for q in NB[p] if b[q]==col]
                if not own: continue
                strong=any(len(group(b,q)[1])>=3 for q in own)
                weakY=any(e(b,q)<=2 for q in own)
                if strong or not weakY: continue
                oppadj=[q for q in NB[p] if b[q]==3-col]
                real=any(len(group(b,q)[1])==1 for q in oppadj)
                flag2=any(len(group(b,q)[1])==2 for q in oppadj)
                cnt[('realcap' if real else ('opp2lib' if flag2 else 'other'))]+=1
print(cnt)
```