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
tot=np.linalg.norm(F)**2
for i in range(6):
    q,k,v,o=hw(0,i); U,S,Vt=np.linalg.svd(o@v)
    r=25 if i==2 else 3
    Oi=U[:,:r]
    # energy of MLP0 read times head output magnitude
    print('head',i,'MLP0 energy frac in its output space (%d d): %.3f'%(r,np.linalg.norm(F@Oi)**2/tot), ' weighted by OV: %.1f'%np.linalg.norm(F@(o@v)))
q,k,v,o=hw(0,2); A=q.T@k; u,s,vt=np.linalg.svd(A)
P=orth(np.hstack([u[:,:18],vt[:18].T]),20)
print('MLP0 energy in P (20d): %.3f'%(np.linalg.norm(F@P)**2/tot))
# How strongly does MLP0 output get read by later layers? proj norms
```

````