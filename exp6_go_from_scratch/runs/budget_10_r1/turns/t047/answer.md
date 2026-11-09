```python
exec(open('/work/an.py').read())
SHAPES_=SHAPES
exec(open('/work/build2.py').read())
A,X,lg=load_pass(6)
P=PV2; K=P['K']
mx=0
for i in range(len(X)):
    b,me=board_from_input(X[i]); op=3-me
    own=(b==me).astype(float); opp=(b==op).astype(float)
    infl=np.zeros(81)
    for p in range(81):
        r,c=divmod(p,9)
        for q in range(81):
            r2,c2=divmod(q,9); d2=(r-r2)**2+(c-c2)**2
            if d2 and d2 in K: infl[p]+=K[d2]*(own[q]-opp[q])
    emp=(b==0)
    ta=np.where(emp,np.maximum(0,P['T']-infl),0); tb=np.where(emp,np.maximum(0,-infl),0)
    gain=np.zeros(81)
    for p in range(81):
        r,c=divmod(p,9)
        for q in range(81):
            r2,c2=divmod(q,9); d2=(r-r2)**2+(c-c2)**2
            if d2 in P['K2']: gain[p]+=P['K2'][d2]*P['alpha']*(ta[q]-P['gamma']*tb[q])
    mx=max(mx,np.abs(A[6][i][387:468]-(gain+P['C'])).max())
print('max G err',mx)
```