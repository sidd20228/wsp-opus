```python
SHAPES_=SHAPES
exec(open('/work/build.py').read())
P2=dict(P0); P2.update(bias=-2.0,linebias={0:-2.5,1:-1.0,2:0.0,3:0.0,4:0.2},alpha=5.0,gamma=1.5,bpass=-9.0,
                     wcap=10.0,woa=8.0,wsa=12.0,wz1=3.0)
import pickle
pickle.dump(P2,open('/work/P2.pkl','wb'))
W=build(P2)
for n in NAMES: save(n,W[n])
print('ok', 'EYE' in open('/work/build.py').read(), W['out.W'][243:250,0:10].min())
```