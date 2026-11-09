All losses are as Black by small margins (−0.5 to −6.5) — Black wins on the board but not by komi. Let me check whether endgame points are being left on the table (unfilled dame, unresolved regions) in those final positions.

```python
exec(open('/work/go.py').read())
import json, numpy as np
G=json.load(open('/work/pass_14/games.json')); B=np.load('/work/pass_14/boards.npy')
def tob(arr): return [1 if x==1 else (2 if x==-1 else 0) for x in arr]
for gi,g in enumerate(G):
    if g['result']!=-1: continue
    n=len(g['moves']); b=tob(B[gi,n-1])
    # regions
    seen=set(); dame=[]; 
    for p in range(81):
        if b[p]==0 and p not in seen:
            reg={p}; st=[p]; border=set()
            while st:
                x=st.pop()
                for y in NB[x]:
                    if b[y]==0 and y not in reg: reg.add(y); st.append(y)
                    elif b[y]: border.add(b[y])
            seen|=reg
            if border=={1,2} or not border: dame.append(sorted(divmod(x,9) for x in reg))
    # who passed first in final sequence
    last=[(m['by'][0], m['move']) for m in g['moves'][-6:]]
    print(gi,'margin',g['margin'],'score(B-W-komi)',score(b),'neutral regions',dame,'last',last)
    for s in board_strings(b): print('    ',s)
```