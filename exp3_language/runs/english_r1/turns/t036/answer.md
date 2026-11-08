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
q,k,v,o=hw(0,2); OV2=o@v; A2=q.T@k
u,s,vt=np.linalg.svd(A2); P=orth(np.hstack([u[:,:20],vt[:20].T]),24)
fc=W['W_fc.0']*W['ln2_w.0']; alive=np.linalg.norm(fc,axis=1)>1.0
R=np.vstack([fc[alive], W['W_v.1']*W['ln1_w.1'], W['W_q.1']*W['ln1_w.1']])
G=R@OV2
Ug,Sg,Vg=np.linalg.svd(G)
print('sv',(Sg[:32]/Sg[0]).round(3))
print('input dirs: energy in P', [round(float(np.linalg.norm(P.T@Vg[r])**2),2) for r in range(32)])
# also how the input dirs relate to MLP0 direct reading (current token read): fc0 reading of these input directions directly
Fd=fc[alive]
print('direct fc0 read of these dirs', [round(float(np.linalg.norm(Fd@Vg[r])),1) for r in range(32)])
```

````