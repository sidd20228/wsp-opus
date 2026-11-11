```python
import numpy as np
fro=np.linalg.norm
H=6;hd=64
def head(l,j):
    g=W[f'ln1_w.{l}'];s=slice(j*hd,(j+1)*hd)
    return W[f'W_q.{l}'][s]*g, W[f'W_k.{l}'][s]*g, W[f'W_v.{l}'][s]*g, W[f'W_o.{l}'][:,s]
OVs=[head(0,j)[3]@head(0,j)[2] for j in range(H)]
U,s,_=np.linalg.svd(np.hstack([OVs[1],OVs[2],OVs[4],OVs[5]]),full_matrices=False)
P=U[:,:2]  # 2D position code (w, out1)
for l in range(1,4):
    row=[]
    for j in range(H):
        Q,K,V,O=head(l,j);M=Q.T@K
        sens=np.linalg.svd(M@P,compute_uv=False)[0]   # max score change per unit of the 2D code, per unit query
        b=W[f'ln1_b.{l}'];bias_sens=fro((Q@b)@K@P)
        row.append('H%d %.2f (bias %.2f)'%(j,sens,bias_sens))
    print('L%d key-position sensitivity:'%l,' '.join(row))
```