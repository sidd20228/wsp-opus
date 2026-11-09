```python
import json, numpy as np
exec(open('/work/an.py').read())
G=json.load(open('/work/pass_2/games.json'))
B=np.load('/work/pass_2/boards.npy'); Vd=np.load('/work/pass_2/valid.npy')
def pb(bb):
    for r in range(9): print(''.join({0:'.',1:'X',-1:'O'}[int(v)] for v in bb[r*9:r*9+9]))
cnt=0
for gi,g in enumerate(G):
    if g['opponent']=='strong' and g['result']<0 and cnt<3:
        cnt+=1
        print(gi,g['net_color'],g['margin'],len(g['moves']),'cap',g['stones_captured_by_net'],'lost',g['stones_lost_by_net'])
        last=int(Vd[gi].sum())-1
        pb(B[gi][last])
        opp_pass=[m['ply'] for m in g['moves'] if m['by']=='opponent' and m['move']==81]
        net_pass=[m['ply'] for m in g['moves'] if m['by']=='net' and m['move']==81]
        print('net passes',net_pass,'opp passes',opp_pass)
print([ (g['net_color'][0],g['margin']) for g in G if g['opponent']=='strong'])
```