```python
import json
d=json.load(open('/work/tests1.json'))
R=[]
for name,m in d['R']:
    m=list(m)
    while m and m[-1]==81: m.pop()
    R.append((name,m))
d['R']=R; json.dump(d,open('/work/tests1.json','w'))
pos=[{"board":b,"to_move":tm} for (_,b,tm) in d['T']]+[{"moves":[("pass" if x==81 else x) for x in m]} for (_,m) in R]
print(json.dumps({"positions":pos,"activations":True}, separators=(',',':')))
```