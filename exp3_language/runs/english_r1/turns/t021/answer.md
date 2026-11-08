<invoke name="python">
```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
H=6; hd=64
def hw(l,j):
    w=W[f'ln1_w.{l}']; s=slice(j*hd,(j+1)*hd)
    return W[f'W_q.{l}'][s]*w, W[f'W_k.{l}'][s]*w, W[f'W_v.{l}'][s]*w, W[f'W_o.{l}'][:,s]
def orth(M,r):
    u,s,vt=np.linalg.svd(M,full_matrices=False); return u[:,:r]
fc=W['W_fc.0']*W['ln2_w.0']
alive=np.linalg.norm(fc,axis=1)>1.0
F=fc[alive]
# output spaces of all L0 heads (what attention writes)
outs=[]
for i in range(6):
    q,k,v,o=hw(0,i); U,S,Vt=np.linalg.svd(o@v)
    r=30 if i==2 else 4
    outs.append(U[:,:r])
Oall=orth(np.hstack(outs),50)
print('F sv',np.linalg.svd(F,compute_uv=False)[:60].round(1))
Fr=F-(F@Oall)@Oall.T
print('F minus head-outputs sv',np.linalg.svd(Fr,compute_uv=False)[:60].round(1))
# what fraction of F energy in head-output space
print('energy frac in head outputs: %.2f'%(np.linalg.norm(F@Oall)**2/np.linalg.norm(F)**2))
# embedding input space: union of L0 attention read spaces
w=W['ln1_w.0']
R=np.vstack([W['W_q.0']*w,W['W_k.0']*w,W['W_v.0']*w])
u,s,vt=np.linalg.svd(R); E=vt[:60].T
print('F-res energy in L0-attn read space(60d): %.2f'%(np.linalg.norm(Fr@E)**2/np.linalg.norm(Fr)**2))
```

````