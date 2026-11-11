Key finding: in most score losses my net was ahead on Tromp-Taylor area when it first passed (often +5 to +15). It then lost 5–20 points while the opponent kept playing and it kept passing. Let me look at what the opponent did in that phase.

```python
import json, numpy as np
exec(open('/work/build8.py').read().split("CC,F0O")[0])
d='/work/pass_9/'
G=json.load(open(d+'games.json')); B=np.load(d+'boards.npy')
def bs(b):
    m={0:'.',1:'X',-1:'O'}
    return [''.join(m[int(b[r*9+c])] for c in range(9)) for r in range(9)]
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
for gi in [3,49]:
    g=G[gi]; me=1 if g['net_color']=='black' else -1; mv=g['moves']
    fp=[m['ply'] for m in mv if m['by']=='net' and m['move']==81][0]
    print('game',gi,g['net_color'])
    print('\n'.join(bs(B[gi,fp-1])))
    for m in mv[fp-2:]:
        j=m['ply']; dd=area(B[gi,j],me)-area(B[gi,j],-me)
        print(j,m['by'],divmod(m['move'],9) if m['move']<81 else 'pass','diff',dd, m['top5'][:3] if m.get('top5') else '')
```