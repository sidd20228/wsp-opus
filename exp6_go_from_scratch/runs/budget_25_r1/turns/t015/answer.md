Margins are bimodal: crushing wins of 70–88 points, plus a few close losses and some with 60–70 of my stones lost. Let me see how those big groups died.

```python
import json, numpy as np
d='/work/pass_2/'
G=json.load(open(d+'games.json')); B=np.load(d+'boards.npy')
def bs(b):
    m={0:'.',1:'X',-1:'O'}
    return [''.join(m[int(b[r*9+c])] for c in range(9)) for r in range(9)]
for gi in [69,77]:
    g=G[gi]; me=1 if g['net_color']=='black' else -1
    mv=g['moves']
    for k in range(1,len(mv)):
        lost=int(((B[gi,k-1]==me)&(B[gi,k]==0)).sum())
        if lost>=5:
            print('game',gi,'ply',k,'opp move',divmod(mv[k]['move'],9),'lost',lost)
            for j in [k-3,k-2,k-1]:
                m=mv[j]; print(' ply',j,m['by'],divmod(m['move'],9) if m['move']<81 else 'pass',m.get('top5'))
            print('\n'.join(bs(B[gi,k-2])))
            print()
            print('\n'.join(bs(B[gi,k])))
            break
```