The cause: an opponent group with 2 far-apart liberties has no local witness, so it looks like it's in atari; the "capture" there is really a 2-stone self-atari, which then leads to the send-two-return-one cycle. Let me measure how often a capture-looking point that is also a self-atari point is a real capture vs. a false one.

```python
exec(open('/work/ana.py').read())
from collections import Counter
G=json.load(open('/work/pass_5/games.json')); Bd=np.load('/work/pass_5/boards.npy')
def tob(arr): return [1 if x==1 else (2 if x==-1 else 0) for x in arr]
def star_pl(b,v):
    c=b[v]; return sum(e(b,q) for q in [v]+NB[v] if b[q]==c)
cnt=Counter()
for gi,g in enumerate(G):
    for k in range(10,len(g['moves']),3):
        b=tob(Bd[gi,k]); col=1 if k%2==1 else 2  # side to move after ply k (ply 0 black)
        ginfo={}
        for p in range(81):
            if b[p] and p not in ginfo:
                gg,l=group(b,p); 
                info=(len(l), any(hv_witness(b,v) for v in gg), any(star_pl(b,v)>=3 for v in gg), len(gg))
                for v in gg: ginfo[v]=info
        for p in range(81):
            if b[p]: continue
            opp_adj=[ginfo[q] for q in NB[p] if b[q]==3-col]
            flagged=[x for x in opp_adj if not x[1]]
            if not flagged: continue
            eps=sum(1 for q in NB[p] if b[q]==0)
            own3=any(ginfo[q][2] for q in NB[p] if b[q]==col)
            sa=(eps<=1 and not own3)
            real=any(x[0]==1 for x in flagged)
            size=max(x[3] for x in flagged if x[0]==1) if real else 0
            cnt[('SA' if sa else 'noSA', 'real' if real else 'false', min(size,3))]+=1
for k in sorted(cnt): print(k,cnt[k])
```