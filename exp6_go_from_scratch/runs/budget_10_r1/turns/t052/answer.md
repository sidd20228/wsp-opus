```python
src=open('/work/build2.py').read()
src=src.replace('''        # EYE = relu(nown - deg + 1)
        b6[EYE+p]+=1-deg[p]
        for q in NB[p]: add_own(q,EYE+p,1)''','''        # Z0 = relu(1 - 3E)  (E==0 indicator)
        b6[EYE+p]+=1
        for q in NB[p]: add_emp(q,EYE+p,-3)''')
open('/work/build3.py','w').write(src)
SHAPES_=SHAPES
exec(src)
PV3=dict(PV2); PV3.update(wcap=40.0,wsa=6.0,wz1=1.5,weye=20.0,bpass=-12.0)
import pickle; pickle.dump(PV3,open('/work/PV3.pkl','wb'))
W=build2(PV3)
for n in NAMES: save(n,W[n])
print('saved', W['L6.b'][306:310], W['out.W'][306,0])
```