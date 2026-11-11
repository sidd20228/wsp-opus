The pattern is almost always the lone stone sitting along the edge next to e (73 of 74). I'll build one unit per edge point covering both along-edge directions. It fires when the inward neighbour is an opponent stone, one along-edge neighbour is my lone stone and the other is opponent or empty. It is weighted so that connecting through a point with an empty neighbour, or connecting a non-lone chain, never fires. There's no capture exemption: these shapes are self-ataris or fake captures far more often than real ones. I'll use the 16 most frequent edge points, then run a paired test against v8E.

```python
src=open('/work/build8e.py').read()
src=src.replace("""    Q['_corn']=CORN""","""    for e in Q['edge_pts']:
        if u>=512: break
        along=[x for x in NB[e] if line(x)==0]; D=[x for x in NB[e] if line(x)!=0][0]
        for x in along:
            W[OWN+x,u]+=2; W[OPP+x,u]+=1
            for q in NB[x]:
                if q!=e: W[OWN+q,u]-=3
        W[OPP+D,u]+=2; W[OWN+D,u]-=4
        b[u]=-4
        CORN.append((u,e)); u+=1
    Q['_corn']=CORN""")
open('/work/build8g.py','w').write(src)
exec(src)
edge_pts=[45*0+53,54,4,36,75,74,18,26,2,78,5,79,63,77,27,44]
print([divmod(x,9) for x in edge_pts], all(line(x)==0 for x in edge_pts), len(set(edge_pts)))
PE=dict(t1=2.5,W_o=0.3,prior={0:-2.0,1:1.3,2:1.5,3:0.7,4:0.9},prior_late={0:-1.0,1:0.4,2:0.55,3:0.35,4:0.35})
PG=dict(PE); PG['edge_pts']=edge_pts
T=build(PG)
exec(open('/work/build8e.py').read())
TE=build(PE)
print({k:float(np.abs(T[k]-TE[k]).max()) for k in T})
print('units used', int((np.abs(T['L6.W']).sum(0)>0).sum()+ (np.abs(T['L6.b'])>0)[ (np.abs(T['L6.W']).sum(0)==0)].sum()))
for k,v in T.items(): save(k,v)
print('saved v8G')
```