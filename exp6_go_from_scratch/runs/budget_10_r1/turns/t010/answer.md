```python
exec(open('/work/an.py').read())
A,X,lg=load_pass(1)
for i in (27,12,32):
    b,me=board_from_input(X[i]); print(i,'me','XO'[me-1]); show(b)
    L=lg[i]; order=np.argsort(-L)[:6]
    print([(divmod(int(o),9) if o<81 else 'pass', round(float(L[o]),2)) for o in order])
    # tents for empty points
    ta=A[6][i][324:405]; tb=A[6][i][405:486]
    print('max TA',ta.max(),'TB at max', [ (divmod(p,9),round(float(ta[p]),2),round(float(tb[p]),2)) for p in np.argsort(-ta)[:5]])
```