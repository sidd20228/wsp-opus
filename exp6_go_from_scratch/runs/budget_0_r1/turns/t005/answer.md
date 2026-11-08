```python
import numpy as np
ns={}; exec(open('/work/build.py').read(), ns)
layers,OW,Ob = ns['build'](ns['P0'], inp)
L5,L6 = layers[4],layers[5]
inv5 = {v:k for k,v in L5.idx.items()}
for key in [('Cap',40),('Save',0),('Pos',40)]:
    j=L6[key]; nz=np.nonzero(L6.W[:,j])[0]
    print(key, 'bias', L6.b[j], [(inv5[i], L6.W[i,j]) for i in nz][:14])
print('out.b sample', Ob[[0,10,20,30,40,81]])
print('out col 40 nnz from:', [ {v:k for k,v in L6.idx.items()}[i] for i in np.nonzero(OW[:,40])[0]])
```