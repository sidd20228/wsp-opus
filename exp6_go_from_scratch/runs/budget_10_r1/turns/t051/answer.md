```python
import json, numpy as np
G=json.load(open('/work/pass_7/games.json'))
B=np.load('/work/pass_7/boards.npy'); Vd=np.load('/work/pass_7/valid.npy')
def pb(bb):
    for r in range(9): print(''.join({0:'.',1:'X',-1:'O'}[int(v)] for v in bb[r*9:r*9+9]))
k=0
for gi,g in enumerate(G):
    if g['opponent']=='strong' and g['end']=='two passes' and g['result']<0 and k<4:
        k+=1; n=int(Vd[gi].sum()); netc=1 if g['net_color']=='black' else -1
        print(gi,g['net_color'],n,g['margin'],'lost',g['stones_lost_by_net'],'cap',g['stones_captured_by_net'])
        prev=np.zeros(81); ev=[]
        for ply in range(n):
            cur=B[gi][ply]; lost=int(((prev==netc)&(cur==0)).sum()); gained=int(((prev==-netc)&(cur==0)).sum())
            if lost>=3: ev.append(('L',ply,lost))
            if gained>=3: ev.append(('C',ply,gained))
            prev=cur
        print(ev); pb(B[gi][n-1])
```