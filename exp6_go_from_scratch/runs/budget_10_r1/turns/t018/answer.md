```python
SHAPES_=SHAPES
exec(open('/work/build.py').read())
src=open('/work/build.py').read()
src=src.replace("Wo[TB+q,p]+=-2*P['alpha']*P['K2'][d2]","Wo[TB+q,p]+=-P['gamma']*P['alpha']*P['K2'][d2]")
open('/work/build.py','w').write(src)
exec(src)
P1=dict(P0); P1.update(bias=-2.0,linebias={0:-2.5,1:-1.0,2:0.0,3:0.0,4:0.2},alpha=5.0,gamma=1.5,bpass=-9.0,wcap=10.0,woa=8.0,wsa=12.0,weye=20.0)
json.dump({k:(v if not isinstance(v,dict) else {str(a):b for a,b in v.items()}) for k,v in P1.items()},open('/work/P1.json','w'))
W=build(P1)
for n in NAMES: save(n,W[n])
print('ok', W['out.b'][[0,10,20,30,40,81]])
```