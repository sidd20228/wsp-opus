```python
src=open('/work/build.py').read()
old_L5=src[src.index('    # L5'):src.index('    # L6')]
new_L5=r'''    # L5
    V5=0; O5=81; CE=162; AM=306; BM=387
    EDGES=[(a,b) for a in range(81) for b in NB[a] if a<b]
    L5=W['L5.W']; b5=W['L5.b']
    for p in range(81):
        L5[V+p,V5+p]=1
        L5[V+p,O5+p]=1; b5[O5+p]=-1
        cm=idm[p]
        for q in NB[p]:
            L5[MX+q,AM+p]+=1; L5[MY+q,AM+p]+=-cm
            L5[MX+q,BM+p]+=-1; L5[MY+q,BM+p]+=cm
    for e,(a,bb) in enumerate(EDGES):
        u=CE+e; b5[u]=1
        for s,t in ((a,bb),(bb,a)):
            c=ido[t]
            L5[OQ+s,u]+=-1; L5[OX+s,u]+=2*c; L5[OY+s,u]+=-c*c
        M=2.0
        L5[V+a,u]+=M; L5[V+bb,u]+=M; b5[u]+=-3*M
    for j in range(4): L5[PR+j,486+j]=1
'''
src=src.replace(old_L5,new_L5)
src=src.replace('''        # CAP = relu(4T + nopp - 4)
        L6[T5+p,CAP+p]=4; b6[CAP+p]=-4
        for q in NB[p]: add_opp(q,CAP+p,1)
''','''        # CAP = relu(sum of capture-edge flags at p)
        for e,(a,bb) in enumerate(EDGES):
            if p in (a,bb): L6[CE+e,CAP+p]=1
''')
open('/work/build.py','w').write(src)
SHAPES_=SHAPES
exec(src)
W=build(P0)
for n in NAMES: save(n,W[n])
print(len(EDGES), 'saved')
```