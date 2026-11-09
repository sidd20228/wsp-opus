```python
import numpy as np
hd=64
Wv=W['W_v.0']
U,S,Vt=np.linalg.svd(Wv)
T=Vt[:30].T
lw=W['ln1_w.0']; lb=W['ln1_b.0']
for j in [0,2,3,4,1,5]:
    s=slice(j*hd,(j+1)*hd)
    Wq=W['W_q.0'][s]; Wk=W['W_k.0'][s]
    QK=Wq.T@Wk
    M=T.T@QK@T
    sym=(M+M.T)/2; asym=(M-M.T)/2
    ev=np.linalg.eigvalsh(sym)
    sv=np.linalg.svd(M,compute_uv=False)
    print(j,'||M|| %.2f sym frac %.2f'%(np.linalg.norm(M),np.linalg.norm(sym)**2/np.linalg.norm(M)**2),'sv',np.round(sv[:6],2),'symeig',np.round(ev[:3],2),np.round(ev[-4:],2))
    # bias-driven key preference: qb = Wq @ lb ; key pref vector in h space = Wk^T qb
    qb=Wq@lb
    kp=Wk.T@qb
    print('   key-pref norm %.2f, frac in T %.2f'%(np.linalg.norm(kp), np.linalg.norm(T.T@kp)**2/np.linalg.norm(kp)**2))
```