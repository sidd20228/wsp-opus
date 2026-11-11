Every cycle has the same root cause. A move with zero or one empty neighbours connects to one of my chains that has exactly two liberties, which I can't detect. Usually a false "capture" signal lures the net there. Two fixes are needed: a better liberty code for opponent chains and stricter self-atari logic.

I'll measure on the recorded game boards which liberty coloring would misjudge two-liberty chains least often.

```python
import json, numpy as np
from collections import Counter
d='/work/pass_2/'
B=np.load(d+'boards.npy'); V=np.load(d+'valid.npy')
def nbrs(p):
    r,c=divmod(p,9); o=[]
    for dr,dc in((-1,0),(1,0),(0,-1),(0,1)):
        rr,cc=r+dr,c+dc
        if 0<=rr<9 and 0<=cc<9:o.append(rr*9+cc)
    return o
NB=[nbrs(p) for p in range(81)]
def chains(b):
    seen=-np.ones(81,int); out=[]
    for p in range(81):
        if b[p]!=0 and seen[p]<0:
            col=b[p]; st=[p]; seen[p]=len(out); mem=[p]
            while st:
                x=st.pop()
                for q in NB[x]:
                    if b[q]==col and seen[q]<0: seen[q]=len(out); st.append(q); mem.append(q)
            libs=set(q for x in mem for q in NB[x] if b[q]==0)
            out.append((col,mem,libs))
    return out
pairs=Counter(); tot2=0; one=0
for gi in range(B.shape[0]):
    for k in range(162):
        if not V[gi,k]: continue
        b=B[gi,k]
        for col,mem,libs in chains(b):
            if len(libs)==2:
                tot2+=1
                if all(sum(b[q]==0 for q in NB[x])<=1 for x in mem):
                    e,f=sorted(libs)
                    pairs[(e,f)]+=1
            if len(libs)==1: one+=1
print('2-lib chains',tot2,'with each stone<=1 lib',sum(pairs.values()),'atari chains',one)
def coll(fn):
    return sum(v for (e,f),v in pairs.items() if fn(e)==fn(f))
codes={'parity':lambda p:(p//9+p%9)%2,
 'r+c mod3':lambda p:(p//9+p%9)%3,'r-c mod3':lambda p:(p//9-p%9)%3,'r+2c mod3':lambda p:(p//9+2*(p%9))%3,
 'r mod2,c mod2 (4)':lambda p:(p//9)%2*2+(p%9)%2,'r+2c mod5':lambda p:(p//9+2*(p%9))%5}
for k,f in codes.items(): print(k,coll(f))
dv=Counter()
for (e,f),v in pairs.items():
    dv[(abs(e//9-f//9),abs(e%9-f%9))]+=v
print(dv.most_common(15))
```