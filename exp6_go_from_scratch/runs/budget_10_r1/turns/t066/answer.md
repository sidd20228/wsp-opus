```python
src=open('/work/build.py').read()
i=src.index('        # OA = relu(nown - 5(AM+BM))'); j=src.index('        # tents: infl')
new='''        cesum=[e for e,(a,bb) in enumerate(EDGES) if p in (a,bb)]
        if P.get('mode','p3')=='p3':
            # OA = relu(nown - 5(AM+BM))
            L6[AM+p,OA+p]=-5; L6[BM+p,OA+p]=-5
            for q in NB[p]: add_own(q,OA+p,1)
        else:
            # P1u = relu(2 - nown - 3E - 9*sumCE)   (E0 & nown<=1, non-capture)
            b6[OA+p]+=2
            for q in NB[p]: add_own(q,OA+p,-1); add_emp(q,OA+p,-3)
            for e in cesum: L6[CE+e,OA+p]+=-9
        # SA = relu(2 - E - 3(AM+BM))
        b6[SA+p]+=2; L6[AM+p,SA+p]=-3; L6[BM+p,SA+p]=-3
        for q in NB[p]: add_emp(q,SA+p,-1)
        # EYE = relu(nown - deg + 1)
        b6[EYE+p]+=1-deg[p]
        for q in NB[p]: add_own(q,EYE+p,1)
'''
src=src[:i]+new+src[j:]
src=src.replace("Wo[CAP+p,p]=P['wcap']; Wo[OA+p,p]=P['woa']; Wo[SA+p,p]=-P['wsa']; Wo[EYE+p,p]=-P['wz1']",
 "Wo[CAP+p,p]=P['wcap']; Wo[OA+p,p]=(P['woa'] if P.get('mode','p3')=='p3' else -P['wp1']); Wo[SA+p,p]=-P['wsa']; Wo[EYE+p,p]=-P['weye']")
open('/work/build_p3.py','w').write(src)
SHAPES_=SHAPES
exec(src)
P3=dict(P0); P3.update(bias=-2.0,linebias={0:-2.5,1:-1.0,2:0.0,3:0.0,4:0.2},alpha=5.0,gamma=1.5,bpass=-9.0,
                     wcap=10.0,woa=8.0,wsa=12.0,weye=20.0,mode='p3')
import pickle; pickle.dump(P3,open('/work/P3final.pkl','wb'))
Wp3=build(P3)
np.savez('/work/p3_weights.npz',**Wp3)
PV=dict(P3); PV.update(mode='v2p',wp1=15.0)
pickle.dump(PV,open('/work/PV2p.pkl','wb'))
Wv=build(PV)
np.savez('/work/v2p_weights.npz',**Wv)
diff={n:float(np.abs(Wp3[n]-Wv[n]).max()) for n in NAMES}
print(diff)
print('TB weight check', Wp3['out.W'][405,0:3], 'SA b', Wp3['L6.b'][162:165], 'EYE b', Wp3['L6.b'][243:246])
```