v14 scores 69% against strong and 77% against weak, and repetitions fell from 16 to 5. Results seem to plateau around 70/75. Let me check the remaining repetitions and the score losses.

```python
import json, numpy as np
from collections import Counter
exec(open('/work/build14.py').read().split("CC,F0O")[0])
d='/work/pass_14/'
G=json.load(open(d+'games.json')); B=np.load(d+'boards.npy')
def chain_of(b,p):
    col=b[p]; mem=[p]; s={p}; st=[p]
    while st:
        x=st.pop()
        for q in NB[x]:
            if b[q]==col and q not in s: s.add(q); st.append(q); mem.append(q)
    libs=set(q for x in mem for q in NB[x] if b[q]==0)
    return mem,libs
def area(b,col):
    s=int((b==col).sum()); seen=np.zeros(81,bool)
    for p in range(81):
        if b[p]==0 and not seen[p]:
            st=[p]; seen[p]=True; mem=[p]; bd=set()
            while st:
                x=st.pop()
                for q in NB[x]:
                    if b[q]==0 and not seen[q]: seen[q]=True; st.append(q); mem.append(q)
                    elif b[q]!=0: bd.add(int(b[q]))
            if bd=={col}: s+=len(mem)
    return s
for gi,g in enumerate(G):
    if g['result']>0: continue
    mv=g['moves']; n=len(mv); me=1 if g['net_color']=='black' else -1
    s=f"{gi} {g['opponent'][:2]} {g['net_color'][0]} {g['end']} mg {g['margin']} len {n} cap {g['stones_captured_by_net']} lost {g['stones_lost_by_net']}"
    if 'rep' in g['end']:
        last=B[gi,n-1]; k=[j for j in range(n-1) if (B[gi,j]==last).all()][-1]
        seq=[(m['by'][0],m['move'],(m['top5'][0][1] if m.get('top5') else None)) for m in mv[k+1:]]
        s+=' seq '+str(seq)
    else:
        fp=[m['ply'] for m in mv if m['by']=='net' and m['move']==81]
        if fp:
            k=fp[0]; s+=f" firstpass {k} diff {area(B[gi,k],me)-area(B[gi,k],-me)} final {area(B[gi,n-1],me)-area(B[gi,n-1],-me)}"
    print(s)
```