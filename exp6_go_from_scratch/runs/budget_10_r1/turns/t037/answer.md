```python
src=open('/work/build.py').read()
old=src[src.index('        # SA = relu(2 - E - 3(AM+BM) - 2*sumCE)'):src.index('        # Z1 = relu')]
new='''        # SA = relu(2 - E - 3(AM+BM) - 2*sumCE + kappa*(sumV - 2deg + 1))
        b6[SA+p]+=2; L6[AM+p,SA+p]=-3; L6[BM+p,SA+p]=-3
        for q in NB[p]: add_emp(q,SA+p,-1)
        for e in cesum: L6[CE+e,SA+p]+=-2
        kap=P.get('kappa',1.0)
        for q in NB[p]: L6[V5+q,SA+p]+=kap
        b6[SA+p]+=kap*(1-2*deg[p])
'''
src=src.replace(old,new)
open('/work/build.py','w').write(src)
SHAPES_=SHAPES
exec(src)
import pickle
P3=pickle.load(open('/work/P2.pkl','rb')); P3.update(wcap=10.0,woa=5.0,wsa=25.0,wz1=6.0,kappa=1.0)
pickle.dump(P3,open('/work/P3.pkl','wb'))
W=build(P3)
for n in NAMES: save(n,W[n])
print('ok')
```