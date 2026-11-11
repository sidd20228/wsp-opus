Again, in most score losses my net was ahead on area at its first pass (e.g. +19, +15, +11) and then collapsed (−33, −30, −18). Game 11 lost 40 stones. Let me trace that.

```python
import json, numpy as np
exec(open('/work/build14.py').read().split("CC,F0O")[0])
d='/work/pass_14/'
G=json.load(open(d+'games.json')); B=np.load(d+'boards.npy')
def bs(b):
    m={0:'.',1:'X',-1:'O'}
    return [''.join(m[int(b[r*9+c])] for c in range(9)) for r in range(9)]
def chains(b):
    lab=-np.ones(81,int); ch=[]
    for p in range(81):
        if b[p]!=0 and lab[p]<0:
            col=b[p]; st=[p]; lab[p]=len(ch); mem=[p]
            while st:
                x=st.pop()
                for q in NB[x]:
                    if b[q]==col and lab[q]<0: lab[q]=len(ch); st.append(q); mem.append(q)
            libs=set(q for x in mem for q in NB[x] if b[q]==0)
            ch.append((col,mem,libs))
    return lab,ch
gi=11; g=G[gi]; mv=g['moves']
print('\n'.join(bs(B[gi,75])))
lab,ch=chains(B[gi,75])
for col,mem,libs in ch:
    if col==1 and len(mem)>=3: print('X chain',len(mem),'libs',sorted(divmod(x,9) for x in libs))
for m in mv[76:]:
    print(m['ply'],m['by'],divmod(m['move'],9) if m['move']<81 else 'pass', m['top5'][:3] if m.get('top5') else '')
```