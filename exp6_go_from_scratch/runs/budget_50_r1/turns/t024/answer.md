SA detection works (96/101 true self-ataris caught, ~3% false alarms). But some self-atari moves still get high logits, and AtO has a notable false-positive rate. Let me inspect those.

```python
exec(open('/work/ana.py').read())
from collections import Counter
d=json.load(open('/work/tests2.json')); lay=json.load(open('/work/layout_v2.json'))
P=load_pass(3)
boards=boards_from_tests({'T':d['T'],'R':d['R']})
for gi,k,p,mv in d['SA']:
    g=Game()
    for x in mv: g.play(x)
    boards.append((list(g.b),g.col))
L5=P['act_L5']; L6=P['act_L6']
# false AtO analysis by group
cnt=Counter()
for i,(b,col) in enumerate(boards):
    seen=set()
    for p in range(81):
        if b[p]==3-col and p not in seen:
            gg,l=group(b,p); seen|=gg
            if len(l)>1 and any(L6[i,lay[5]['AtO']+q]>0.5 for q in gg):
                dist=chain_dist(b,p); ws=[q for q in gg if hv_witness(b,q)]
                cnt[(len(l), 'nowit' if not ws else 'radius')]+=1
print(cnt)
for j in [2,4,9,19,23,6]:
    gi,k,p,mv=d['SA'][j]; i=26+j; b,col=boards[i]
    print('pos',i,'move',divmod(p,9),'col',col)
    for s in board_strings(b): print('   ',s)
    for q in NB[p]:
        print('   nb',divmod(q,9),'val',b[q],'AtO',L6[i,lay[5]['AtO']+q],'AtM',L6[i,lay[5]['AtM']+q],'L2O',L6[i,lay[5]['L2O']+q], 'libs',len(group(b,q)[1]) if b[q] else '-')
```