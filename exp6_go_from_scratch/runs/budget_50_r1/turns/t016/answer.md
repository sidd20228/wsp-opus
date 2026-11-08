Star-based witnesses err rarely (3% of atari groups, 7% of 2-lib groups). Let me compare candidate witness designs on real game positions before restructuring.

```python
exec(open('/work/ana.py').read())
G=json.load(open('/work/pass_2/games.json')); Bd=np.load('/work/pass_2/boards.npy')
def tob(arr): return [1 if x==1 else (2 if x==-1 else 0) for x in arr]
DG=[]
for p in range(81):
    r,c=divmod(p,9); DG.append([(r+dr)*9+c+dc for dr,dc in ((-1,-1),(-1,1),(1,-1),(1,1)) if 0<=r+dr<9 and 0<=c+dc<9])
def star_pl(b,v):
    c=b[v]; return sum(e(b,q) for q in [v]+NB[v] if b[q]==c)
def sh(b,y,c): 
    if b[y]!=0: return 0
    return max(0,sum(1 for q in NB[y] if b[q]==c)-1)
def starcorr(b,v):
    c=b[v]; return star_pl(b,v)-sum(sh(b,y,c) for y in DG[v])
def star_exact(b,v):
    c=b[v]; libs=set()
    for q in [v]+NB[v]:
        if b[q]==c: libs|={y for y in NB[q] if b[y]==0}
    return len(libs)
res=Counter()
for gi,g in enumerate(G):
    for k in range(0,len(g['moves']),7):
        b=tob(Bd[gi,k]); seen=set()
        for p in range(81):
            if b[p] and p not in seen:
                gg,l=group(b,p); seen|=gg; nl=len(l)
                key='1' if nl==1 else ('2' if nl==2 else '3+')
                res[key,'n']+=1
                res[key,'hv2']+=any(hv_witness(b,v) for v in gg)
                res[key,'sc2']+=any(starcorr(b,v)>=2 for v in gg)
                res[key,'st2']+=any(star_pl(b,v)>=2 for v in gg)
                res[key,'ex2']+=any(star_exact(b,v)>=2 for v in gg)
                res[key,'st3']+=any(star_pl(b,v)>=3 for v in gg)
                res[key,'ex3']+=any(star_exact(b,v)>=3 for v in gg)
                res[key,'hvsc2']+=any(hv_witness(b,v) or starcorr(b,v)>=2 for v in gg)
for key in ['1','2','3+']:
    print(key,{k2:res[key,k2] for k2 in ['n','hv2','sc2','st2','ex2','hvsc2','st3','ex3']})
```