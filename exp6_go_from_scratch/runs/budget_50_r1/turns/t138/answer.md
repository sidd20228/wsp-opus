```python
import json
A=json.load(open('/work/pass_33/games.json')); Bv=json.load(open('/work/pass_35/games.json'))
As=[g for g in A if g['opponent']=='strong']; Bs=[g for g in Bv if g['opponent']=='strong']
same=sum(1 for a,b in zip(As,Bs) if [m['move'] for m in a['moves'][:6]]==[m['move'] for m in b['moves'][:6]])
same_shift=sum(1 for a,b in zip(As[2:],Bs) if [m['move'] for m in a['moves'][:6]]==[m['move'] for m in b['moves'][:6]])
print('same first6 (aligned)',same,' (shifted by 2)',same_shift, len(As),len(Bs))
print([ (g['net_color'],g['margin'],g['end']) for g in Bs if g['result']==-1])
```