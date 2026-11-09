```python
exec(open('/work/ana.py').read())
from collections import Counter
d=json.load(open('/work/tests1.json')); lay=json.load(open('/work/layout_v10.json'))
P=load_pass(12); boards=boards_from_tests(d)
L5=P['act_L5']; L6=P['act_L6']
st=Counter()
for i,(b,col) in enumerate(boards):
    for p in range(81):
        if b[p]==0: continue
        gg,l=group(b,p); n=min(len(l),3)
        if b[p]==3-col:
            st['AtO',n,L5[i,lay[4]['AtO']+p]>0.5]+=1; st['L2O',n,L5[i,lay[4]['L2O']+p]>0.5]+=1
        else:
            st['L2M',n,L5[i,lay[4]['L2M']+p]>0.5]+=1
        # empty-point checks
for k in sorted(st): print(k,st[k])
# check TAC consistency for one position
i=4; b,col=boards[i]
print([ (divmod(p,9), round(float(L6[i,lay[5]['TAC']+p]),1)) for p in range(81) if L6[i,lay[5]['TAC']+p]>0.1])
print('activation max per layer',[float(P['act_L%d'%k].max()) for k in range(1,7)])
```