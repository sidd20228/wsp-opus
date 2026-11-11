```python
import numpy as np
fro=np.linalg.norm
Up,sp,_=np.linalg.svd(W['W_proj.3'],full_matrices=False)
Wo=W['W_o.3']
e=np.array([fro(Up[:,i]@Wo) for i in range(384)])
print('W_o.3 random-dir energy baseline %.2f'%(fro(Wo)/np.sqrt(384)))
print('W_o.3 energy along W_proj.3 dirs 0-40:',np.round(e[:40],2))
print('dirs 40-100 mean %.2f, 100-384 mean %.2f'%(e[40:100].mean(),e[100:].mean()))
# same for W_proj.2 and W_o.2 (internal) as comparison
for nm in ['W_proj.2','W_o.2']:
    A=W[nm];ee=np.array([fro(Up[:,i]@A) for i in range(384)])
    print(nm,'baseline %.2f'%(fro(A)/np.sqrt(384)),'dirs0-32',np.round(ee[:32],2),' rest mean %.2f'%ee[40:].mean())
```