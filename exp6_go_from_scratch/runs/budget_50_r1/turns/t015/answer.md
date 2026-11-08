The cycles are "send-two-return-one": my net extends a stone that has 2 liberties into a self-atari (ε(p)=0, connecting to a 2-liberty group — a case my self-atari detector can't see), the opponent captures two, and my recapture repeats the position. Let me quantify self-atari moves and how often a cheaper "star" witness would err on real game positions.

```python
exec(open('/work/ana.py').read())
G=json.load(open('/work/pass_2/games.json')); Bd=np.load('/work/pass_2/boards.npy')
def tob(arr): return [1 if x==1 else (2 if x==-1 else 0) for x in arr]
sa_moves=0; sa_cap=0; net_moves=0; sa_types=Counter() if 'Counter' in dir() else None
from collections import Counter
sa_types=Counter()
star2_miss=0; atari_groups=0; star3_over=0; twolib_groups=0
def star_pl(b,v):
    c=b[v]; return sum(e(b,q) for q in [v]+NB[v] if b[q]==c)
for gi,g in enumerate(G):
    netc=1 if g['net_color']=='black' else 2
    prev=[0]*81
    for k,m in enumerate(g['moves']):
        b=tob(Bd[gi,k])
        if m['by']=='net' and m['move']<81:
            net_moves+=1; p=m['move']
            capt=sum(1 for q in range(81) if prev[q]==3-netc and b[q]==0)
            gg,l=group(b,p)
            if len(l)==1 and capt==0:
                sa_moves+=1
                eps=sum(1 for q in NB[p] if prev[q]==0)
                nown=sum(1 for q in NB[p] if prev[q]==netc)
                sa_types[(eps,nown,len(gg))]+=1
        if k%10==0:
            seen=set()
            for p in range(81):
                if b[p] and p not in seen:
                    gg,l=group(b,p); seen|=gg
                    if len(l)==1:
                        atari_groups+=1
                        if any(star_pl(b,v)>=2 for v in gg): star2_miss+=1
                    if len(l)==2:
                        twolib_groups+=1
                        if any(star_pl(b,v)>=3 for v in gg): star3_over+=1
        prev=b
print('net moves',net_moves,'self-atari non-capture moves',sa_moves)
print(sa_types.most_common(15))
print('atari groups',atari_groups,'star2 false witness',star2_miss,' 2lib groups',twolib_groups,'star3 false',star3_over)
```