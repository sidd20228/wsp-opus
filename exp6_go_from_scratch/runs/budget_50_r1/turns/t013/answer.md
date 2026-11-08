All repetition endings are losses caused by my net — often in games it was winning big. Let me inspect the cycle.

```python
import json, numpy as np
G=json.load(open('/work/pass_2/games.json')); B=np.load('/work/pass_2/boards.npy')
def show(b):
    ch={1:'X',-1:'O',0:'.'}
    return [''.join(ch[int(b[r*9+c])] for c in range(9)) for r in range(9)]
cnt=0
for gi,g in enumerate(G):
    if g['end']=='repetition' and cnt<3:
        cnt+=1
        mv=g['moves']; n=len(mv)
        print('game',gi,g['opponent'],'net',g['net_color'],'margin',g['margin'])
        for k in range(n-7,n):
            m=mv[k]
            print(' ply',m['ply'],m['by'],divmod(m['move'],9) if m['move']<81 else 'pass', m.get('top5',''))
            rows=show(B[gi,k])
            print('   '+' | '.join(rows))
```