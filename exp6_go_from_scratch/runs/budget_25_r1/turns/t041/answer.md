The net passed from ply 86 while the opponent captured its right-edge stones one by one. The saving moves were false-eye connection points, which got the full eye penalty of −50. Let me check the chains and liberties at ply 85.

```python
import json, numpy as np
exec(open('/work/build7.py').read().split("CC,F0O")[0])
d='/work/pass_7/'
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
gi=32
for k in [85,89]:
    b=B[gi,k]; print('after ply',k); print('\n'.join(bs(b)))
    lab,ch=chains(b)
    for col,mem,libs in ch:
        if col==1 and len(libs)<=3: print(' X chain',len(mem),sorted(divmod(x,9) for x in mem)[:6],'libs',sorted(divmod(x,9) for x in libs))
```