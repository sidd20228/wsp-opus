Repetitions still start from fake-capture connections, and overall strength fell sharply. The new SA_conn penalises an E=0 connection between two weak chains at 6 units (−36). The worse-performing v10 also penalised these heavily (5 units), while v8B used only 2. These are typically cut-point connections, which matter a lot, so v8B's combined self-atari stays the best structure. Before deciding where to go next, let me check v8B's losses for points lost while my net passed and the opponent kept playing.

```python
import json, numpy as np
exec(open('/work/build8.py').read().split("CC,F0O")[0])
d='/work/pass_9/'
G=json.load(open(d+'games.json')); B=np.load(d+'boards.npy')
def area(b,col):
    # tromp-taylor area for col
    s=int((b==col).sum())
    seen=np.zeros(81,bool)
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
    if g['result']>0 or 'two' not in g['end']: continue
    me=1 if g['net_color']=='black' else -1
    mv=g['moves']; n=len(mv)
    fp=[m['ply'] for m in mv if m['by']=='net' and m['move']==81]
    if not fp: continue
    k=fp[0]
    def diff(j): return area(B[gi,j],me)-area(B[gi,j],-me)
    print(gi,g['opponent'][:2],g['net_color'][0],'mg',g['margin'],'first net pass ply',k,'of',n,'area diff at first pass',diff(k),'final',diff(n-1),'net passes',len(fp))
```