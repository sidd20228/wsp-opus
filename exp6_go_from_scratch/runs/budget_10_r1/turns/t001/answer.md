```python
import os, numpy as np
print(SHAPES); print(NAMES); print(saved())
print(inp(0,0,0), inp(8,8,2), mv(8,8), PASS)
print(os.listdir('/work'))
# search small modular ids for quadratic test: id=(r+a*k)%m distinct from center within manhattan radius R
def ok(a,m,R=5):
    for dr in range(-8,9):
        for dk in range(-8,9):
            if (dr,dk)==(0,0) or abs(dr)+abs(dk)>R: continue
            if (dr+a*dk)%m==0: return False
    return True
for m in range(5,50):
    sol=[a for a in range(1,m) if ok(a,m)]
    if sol: print("m",m,"a",sol); break
```