```python
import numpy as np
hd=64
Wv=W['W_v.0']
U,S,Vt=np.linalg.svd(Wv)
T=Vt[:30].T  # 384x30 basis (post-LN h space)
P=T@T.T
for j in range(6):
    s=slice(j*hd,(j+1)*hd)
    Wq=W['W_q.0'][s]; Wk=W['W_k.0'][s]; Wvh=W['W_v.0'][s]
    QK=Wq.T@Wk
    fq=np.linalg.norm(P@QK)**2/np.linalg.norm(QK)**2
    fk=np.linalg.norm(QK@P)**2/np.linalg.norm(QK)**2
    fqk=np.linalg.norm(P@QK@P)**2/np.linalg.norm(QK)**2
    fv=np.linalg.norm(Wvh@P)**2/np.linalg.norm(Wvh)**2
    fq_raw=np.linalg.norm(Wq@P)**2/np.linalg.norm(Wq)**2
    fk_raw=np.linalg.norm(Wk@P)**2/np.linalg.norm(Wk)**2
    print(j,'QK: q-side in T %.2f, k-side in T %.2f, both %.2f | Wq in T %.2f Wk in T %.2f Wv in T %.2f'%(fq,fk,fqk,fq_raw,fk_raw,fv))
print('random expectation', 30/384)
# Also MLP
Wfc=W['W_fc.0']; print('Wfc in T', np.linalg.norm(Wfc@P)**2/np.linalg.norm(Wfc)**2)
```