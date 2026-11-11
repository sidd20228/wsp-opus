```python
import numpy as np
fro=np.linalg.norm
H=6;hd=64
def head(l,j):
    g=W[f'ln1_w.{l}'];s=slice(j*hd,(j+1)*hd)
    return W[f'W_q.{l}'][s]*g, W[f'W_k.{l}'][s]*g, W[f'W_v.{l}'][s]*g, W[f'W_o.{l}'][:,s]
Q3,K3,V3,O3=head(0,3);OV3=O3@V3;M3=Q3.T@K3
ev,evec=np.linalg.eig(M3);order=np.argsort(-np.abs(ev))
for k in [10,20,30,40]:
    idx=order[:k];Bp,_=np.linalg.qr(np.hstack([evec[:,idx].real,evec[:,idx].imag]))
    Pr=np.eye(384)-Bp@Bp.T
    s=np.linalg.svd(OV3@Pr,compute_uv=False)
    print('k=%d posdim=%d OV3 w/o pos sv:'%(k,Bp.shape[1]),np.round(s[:34],2))
# also spectrum of combined token read: all L0 V (heads 0,3) and fc0, after removing L0H3 pos planes (k=20)
idx=order[:20];Bp,_=np.linalg.qr(np.hstack([evec[:,idx].real,evec[:,idx].imag]));Pr=np.eye(384)-Bp@Bp.T
s=np.linalg.svd(W['W_v.0']*W['ln1_w.0']@Pr,compute_uv=False);print('W_v.0 w/o pos',np.round(s[:40],2))
```