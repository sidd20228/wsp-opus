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
q,k,v,o=hw(0,2); A2=q.T@k; u,s,vt=np.linalg.svd(A2)
P=orth(np.hstack([u[:,:20],vt[:20].T]),24)
U2,S2,V2=np.linalg.svd(o@v); T=V2[:30].T
PT=orth(np.hstack([P,T]),54)
res={}
for i in [0,1,3,4,5]:
    q,k,v,o=hw(0,i); QK=q.T@k; U,S,Vt=np.linalg.svd(QK)
    for r in range(2):
        x=Vt[r]-PT@(PT.T@Vt[r]); res[(i,r)]=x/np.linalg.norm(x); print(i,r,'rest norm^2 %.2f'%(x@x))
ks=list(res)
print(np.array([[abs(res[a]@res[b]) for b in ks] for a in ks]).round(2))
# also compare with the bias ln1_b.0 direction and the mean of ln1_w
b=W['ln1_b.0']; w=W['ln1_w.0']
for a in ks: print(a,'cos with ln1_b: %.2f'%(res[a]@b/np.linalg.norm(b)))
```

````