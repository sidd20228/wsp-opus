Each of these is an E=0 connection to my own single two-liberty stone next to a big opponent chain. The fake capture claim gives partial exemption plus a discounted bonus. Late in the game that's enough to beat passing, and in game 69 it was even the top move at 0.999. I'll tighten this with a smaller capture exemption, a bigger connectivity discount and a stronger self-atari weight. First, let me check what's left on the board when the close score losses end: dame, and stones in atari.

```python
import json, numpy as np
exec(open('/work/build8.py').read().split("CC,F0O")[0])
d='/work/pass_9/'
G=json.load(open(d+'games.json')); B=np.load(d+'boards.npy')
def regions(b):
    seen=np.zeros(81,bool); out=[]
    for p in range(81):
        if b[p]==0 and not seen[p]:
            st=[p]; seen[p]=True; mem=[p]; border=set()
            while st:
                x=st.pop()
                for q in NB[x]:
                    if b[q]==0 and not seen[q]: seen[q]=True; st.append(q); mem.append(q)
                    elif b[q]!=0: border.add(int(b[q]))
            out.append((mem,border))
    return out
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
for gi,g in enumerate(G):
    if g['result']>0 or 'two' not in g['end']: continue
    n=len(g['moves']); b=B[gi,n-1]; me=1 if g['net_color']=='black' else -1
    dame=sum(len(m) for m,bd in regions(b) if len(bd)==2)
    lab,ch=chains(b)
    opp_atari=sum(len(m) for c,m,l in ch if c==-me and len(l)==1)
    my_atari=sum(len(m) for c,m,l in ch if c==me and len(l)==1)
    print(gi,g['opponent'][:2],g['net_color'][0],'mg',g['margin'],'dame',dame,'opp stones in atari',opp_atari,'my stones in atari',my_atari)
```