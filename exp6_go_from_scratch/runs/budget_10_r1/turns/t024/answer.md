```python
src=open('/work/build.py').read()
old=src[src.index('        # SA = relu(2 - E - 3(AM+BM))'):src.index('        # tents')]
new='''        cesum=[e for e,(a,bb) in enumerate(EDGES) if p in (a,bb)]
        # SA = relu(2 - E - 3(AM+BM) - 2*sumCE)
        b6[SA+p]+=2; L6[AM+p,SA+p]=-3; L6[BM+p,SA+p]=-3
        for q in NB[p]: add_emp(q,SA+p,-1)
        for e in cesum: L6[CE+e,SA+p]+=-2
        # Z1 = relu(nown - 3E + 3 - 9*sumCE)   (stored in EYE slot)
        b6[EYE+p]+=3
        for q in NB[p]: add_own(q,EYE+p,1); add_emp(q,EYE+p,-3)
        for e in cesum: L6[CE+e,EYE+p]+=-9
'''
src=src.replace(old,new)
src=src.replace("Wo[EYE+p,p]=-P['weye']","Wo[EYE+p,p]=-P['wz1']")
open('/work/build.py','w').write(src)
SHAPES_=SHAPES
exec(src)
P2=dict(P1); P2.update(wz1=3.0)
W=build(P2)
for n in NAMES: save(n,W[n])
json.dump({k:(v if not isinstance(v,dict) else {str(a):b for a,b in v.items()}) for k,v in P2.items()},open('/work/P2.json','w'))
print('ok')
```