<invoke name="python">
```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
U,S,Vt=np.linalg.svd(W['W_proj.3'])
print('W_proj.3 sv 20-60', S[20:60].round(2))
Out=U[:,:29]
for j in range(6):
    s=slice(j*64,(j+1)*64)
    OV=W['W_o.3'][:,s]@(W['W_v.3'][s]*W['ln1_w.3'])
    Uo,So,Vo=np.linalg.svd(OV)
    # energy of OV's output in the W_proj.3 top-29 space
    fr=np.linalg.norm(Out.T@OV)**2/np.linalg.norm(OV)**2
    # per output dir r of W_proj.3, how much does this head write
    per=np.linalg.norm(U[:,:32].T@OV,axis=1)
    print('L3H%d frac of OV energy in proj3-top29: %.2f'%(j,fr), ' per-dir', (per/np.linalg.norm(OV)*np.sqrt(384)).round(1))
# random baseline frac = 29/384
print('baseline', 29/384)
```

````