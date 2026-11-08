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
q,k,v,o=hw(0,2); OV2=o@v
fc=W['W_fc.0']*W['ln2_w.0']; alive=np.linalg.norm(fc,axis=1)>1.0
F=fc[alive]
outs=[np.linalg.svd(hw(0,i)[3]@hw(0,i)[2])[0][:,:(25 if i==2 else 4)] for i in range(6)]
Oall=orth(np.hstack(outs),45)
X=F-(F@Oall)@Oall.T
Uo,So,Vo=np.linalg.svd(OV2); 
Ux,Sx,Vx=np.linalg.svd(X)
for ka,kb in [(30,40),(30,60),(40,60),(35,80)]:
    A=Vo[:ka].T; B=Vx[:kb].T
    c=np.linalg.svd(A.T@B,compute_uv=False)
    print(ka,kb,c.round(2))
```

````