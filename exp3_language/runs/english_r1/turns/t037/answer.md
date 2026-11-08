<invoke name="python">
```python
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=220)
H=6; hd=64
def hw(l,j):
    w=W[f'ln1_w.{l}']; s=slice(j*hd,(j+1)*hd)
    return W[f'W_q.{l}'][s]*w, W[f'W_k.{l}'][s]*w, W[f'W_v.{l}'][s]*w, W[f'W_o.{l}'][:,s]
def orth(M,r):
    u,s,vt=np.linalg.svd(M,full_matrices=False); return u[:,:r]
q,k,v,o=hw(0,2); OV2=o@v; A2=q.T@k
u,s,vt=np.linalg.svd(A2)
for rp in [0,20,30,40]:
    if rp: P=orth(np.hstack([u[:,:rp],vt[:rp].T]),rp+4); X=OV2-(OV2@P)@P.T
    else: X=OV2
    sv=np.linalg.svd(X,compute_uv=False)
    print(rp,(sv[:34]).round(3))
# compare: fc0 alive direct read, minus P(40) and minus head outputs
fc=W['W_fc.0']*W['ln2_w.0']; alive=np.linalg.norm(fc,axis=1)>1.0
F=fc[alive]
outs=[np.linalg.svd(hw(0,i)[3]@hw(0,i)[2])[0][:,:(25 if i==2 else 4)] for i in range(6)]
Oall=orth(np.hstack(outs),45)
P=orth(np.hstack([u[:,:40],vt[:40].T]),44)
X=F-(F@Oall)@Oall.T; X=X-(X@P)@P.T
print('fc0 direct (minus head outs, minus P44):',np.linalg.svd(X,compute_uv=False)[:40].round(2))
```

````