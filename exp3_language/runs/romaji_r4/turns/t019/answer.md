```python
import numpy as np
fro=np.linalg.norm
H=6;hd=64
def head(l,j):
    g=W[f'ln1_w.{l}'];s=slice(j*hd,(j+1)*hd)
    return W[f'W_q.{l}'][s]*g, W[f'W_k.{l}'][s]*g, W[f'W_v.{l}'][s]*g, W[f'W_o.{l}'][:,s]
Q3,K3,V3,O3=head(0,3);M3=Q3.T@K3
ev,evec=np.linalg.eig(M3);order=np.argsort(-np.abs(ev))[:20]
Bp,_=np.linalg.qr(np.hstack([evec[:,order].real,evec[:,order].imag]))
OV5=head(0,5)[3]@head(0,5)[2];U5,s5,V5=np.linalg.svd(OV5);r=V5[0]
for j in [0,1,2,4,5]:
    Q,K,V,O=head(0,j);u,s,vt=np.linalg.svd(Q.T@K)
    print('L0H%d QK s'%j,np.round(s[:4],2))
    for i in range(3):
        print('   comp%d: q in pos %.2f, k in pos %.2f, q.r %+.2f, k.r %+.2f, q.k %+.2f'%(i,fro(Bp.T@u[:,i])**2,fro(Bp.T@vt[i])**2,u[:,i]@r,vt[i]@r,u[:,i]@vt[i]))
```