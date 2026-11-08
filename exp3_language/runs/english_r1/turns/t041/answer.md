<invoke name="python">
```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
U,S,Vt=np.linalg.svd(W['W_proj.3'])
for r in range(34):
    x=Vt[r]
    # sign-consistency and sparsity (kurtosis)
    m=x.mean()/x.std()*np.sqrt(len(x))/np.sqrt(len(x))
    kurt=((x-x.mean())**4).mean()/x.var()**2
    top=np.sort(np.abs(x))[::-1]
    print(r,'S %.2f'%S[r],' mean/std %.2f'%(x.mean()/x.std()),' kurt %.1f'%kurt,' top10 share %.2f'%((top[:10]**2).sum()),' frac pos %.2f'%((x>0).mean()))
```

````