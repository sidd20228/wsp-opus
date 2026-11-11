```python
import numpy as np
def rows(m,j,l): 
    s=slice(j*64,(j+1)*64); return W[f'W_{m}.{l}'][s]*W[f'ln1_w.{l}']
M=np.vstack([rows('q',2,0),rows('k',2,0)])
u,s,vt=np.linalg.svd(M,full_matrices=False)
print(s[:26].round(2))
P=vt[:20].T  # 384x20
OVb=np.hstack([W['W_o.0'][:,j*64:(j+1)*64]@rows('v',j,0) for j in [0,1,4,5]])
ub,sb,_=np.linalg.svd(OVb,full_matrices=False); Bo=ub[:,:3]
print('overlap P with broad-out',np.linalg.svd(P.T@Bo,compute_uv=False).round(2))
def frac(QK,A,Bm): return np.linalg.norm(A.T@QK@Bm)**2/np.linalg.norm(QK)**2
for l in [0,1,2,3]:
    for j in range(6):
        Wq=rows('q',j,l);Wk=rows('k',j,l);QK=Wq.T@Wk
        Wv=rows('v',j,l)
        print(f'L{l}H{j} QK PP={frac(QK,P,P):.2f} BoBo={frac(QK,Bo,Bo):.2f} anyP-K={np.linalg.norm(QK@P)**2/np.linalg.norm(QK)**2:.2f} anyP-Q={np.linalg.norm(P.T@QK)**2/np.linalg.norm(QK)**2:.2f}  V reads P={np.linalg.norm(Wv@P)**2/np.linalg.norm(Wv)**2:.2f} V reads Bo={np.linalg.norm(Wv@Bo)**2/np.linalg.norm(Wv)**2:.2f}')
```