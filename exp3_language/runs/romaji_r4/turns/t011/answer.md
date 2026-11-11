```python
import numpy as np
fro=np.linalg.norm
H=6;hd=64
def head(l,j):
    g=W[f'ln1_w.{l}'];s=slice(j*hd,(j+1)*hd)
    return W[f'W_q.{l}'][s]*g, W[f'W_k.{l}'][s]*g, W[f'W_v.{l}'][s]*g, W[f'W_o.{l}'][:,s]
Up,sp,_=np.linalg.svd(W['W_proj.3'],full_matrices=False)
for k in [21,24,26,28,32]:
    P=Up[:,:k]
    row=[]
    for j in range(H):
        Q,K,V,O=head(3,j);OV=O@V
        row.append('H%d %.2f'%(j,fro(P.T@OV)**2/fro(OV)**2))
    row.append('W_proj.2 %.2f'%(fro(P.T@W['W_proj.2'])**2/fro(W['W_proj.2'])**2))
    row.append('W_o.2 %.2f'%(fro(P.T@W['W_o.2'])**2/fro(W['W_o.2'])**2))
    print('top-%d W_proj.3 dirs (random %.3f):'%(k,k/384),' '.join(row))
# spectrum of L3 head OVs projected onto top-28 vs the rest
P=Up[:,:28]
for j in range(H):
    Q,K,V,O=head(3,j);s=np.linalg.svd(P.T@O@V,compute_uv=False);print('L3H%d OV in out-space sv'%j,np.round(s[:30],2))
# per-direction write strength of W_proj.3 vs W_o.3 in the top-32
print('W_o.3 energy per W_proj.3 sing dir:',np.round([fro(Up[:,i]@W['W_o.3']) for i in range(32)],2))
```