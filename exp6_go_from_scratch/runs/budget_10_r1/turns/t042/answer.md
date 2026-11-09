```python
def ok(a,m,R):
    for dr in range(-8,9):
        for dk in range(-8,9):
            if (dr,dk)==(0,0) or abs(dr)+abs(dk)>R: continue
            if (dr+a*dk)%m==0: return False
    return True
for m in range(5,60):
    sol=[a for a in range(1,m) if ok(a,m,6)]
    if sol: print("R6 m",m,"a",sol); break
```