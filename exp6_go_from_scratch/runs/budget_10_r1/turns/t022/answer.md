```python
import json, numpy as np
G=json.load(open('/work/pass_3/games.json'))
B=np.load('/work/pass_3/boards.npy'); Vd=np.load('/work/pass_3/valid.npy')
def pb(bb):
    for r in range(9): print(''.join({0:'.',1:'X',-1:'O'}[int(v)] for v in bb[r*9:r*9+9]))
k=0
for gi,g in enumerate(G):
    if g['end']=='two passes' and g['result']<0 and g['opponent']=='strong' and k<3:
        k+=1; n=int(Vd[gi].sum())
        print(gi,g['net_color'],n,'margin',g['margin'],'lost',g['stones_lost_by_net'],'cap',g['stones_captured_by_net'])
        pb(B[gi][n-1])
        # find big capture events against net
        netc=1 if g['net_color']=='black' else -1
        prev=np.zeros(81)
        for ply in range(n):
            cur=B[gi][ply]; lost=int(((prev==netc)&(cur==0)).sum())
            if lost>=4: print('  ply',ply,'net lost',lost)
            prev=cur
```