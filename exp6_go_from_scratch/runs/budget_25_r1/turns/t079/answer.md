v15 scores 74% against strong but 65% against weak. Overall (70% of 128) it is indistinguishable from v14 (72%) given the noise. Tuning single weights won't get past this plateau. The largest consistent loss source is the late-game collapse after my first pass, so let me examine individual opponent moves that cost 3+ area points after it.

```python
import json, numpy as np
exec(open('/work/build14.py').read().split("CC,F0O")[0])
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
d='/work/pass_15/'
G=json.load(open(d+'games.json')); B=np.load(d+'boards.npy')
shown=0
for gi,g in enumerate(G):
    if g['result']>0 or 'two' not in g['end']: continue
    me=1 if g['net_color']=='black' else -1; mv=g['moves']
    fp=[m['ply'] for m in mv if m['by']=='net' and m['move']==81]
    if not fp: continue
    for m in mv[fp[0]:]:
        k=m['ply']
        if m['by']!='opponent' or k<1: continue
        d0=area(B[gi,k-1],me)-area(B[gi,k-1],-me); d1=area(B[gi,k],me)-area(B[gi,k],-me)
        if d0-d1>=4 and shown<4:
            prevnet=mv[k-1]
            print('game',gi,g['net_color'],'ply',k,'opp move',divmod(m['move'],9),'area drop',d0-d1,'our prev',prevnet['move'],prevnet.get('top5','')[:3])
            print('\n'.join(bs(B[gi,k-2])))
            shown+=1
            break
```