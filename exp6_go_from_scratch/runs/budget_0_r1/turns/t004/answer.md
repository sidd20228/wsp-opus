```python
import numpy as np
ns={}
exec(open('/work/build.py').read(), ns)
layers,OW,Ob = ns['build'](ns['P0'], inp)
for k,L in enumerate(layers, start=1):
    save(f'L{k}.W', L.W.astype(np.float32))
    save(f'L{k}.b', L.b.astype(np.float32))
save('out.W', OW.astype(np.float32)); save('out.b', Ob.astype(np.float32))
print(saved())
# structural checks: units beyond n have zero weights/bias, and next layer reads only existing units
for k,L in enumerate(layers):
    assert np.all(L.W[:,L.n:]==0) and np.all(L.b[L.n:]==0)
    if k+1 < len(layers):
        nxt=layers[k+1]; assert np.all(nxt.W[L.n:,:]==0)
assert np.all(OW[layers[-1].n:,:]==0)
print('ok', [load(n).shape for n in NAMES][:3])
```