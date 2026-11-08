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
fc=W['W_fc.0']*W['ln2_w.0']; alive=np.linalg.norm(fc,axis=1)>1.0
F=fc[alive]
U,S,Vt=np.linalg.svd(F,full_matrices=False)
q,k,v,o=hw(0,2); A2=q.T@k; u,s,vt=np.linalg.svd(A2)
P=orth(np.hstack([u[:,:20],vt[:20].T]),24)
U2,S2,V2=np.linalg.svd(o@v); T=V2[:20].T; O2=U2[:,:20]
outs=[np.linalg.svd(hw(0,i)[3]@hw(0,i)[2])[0][:,:3] for i in [0,1,3,4,5]]
Ob=orth(np.hstack(outs),6)
for r in range(4):
    x=Vt[r]
    print(r,'S %.1f'%S[r],' P %.2f T %.2f O2 %.2f Ob %.2f'%tuple(float(np.linalg.norm(Sp.T@x)**2) for Sp in [P,T,O2,Ob]))
# neuron loadings on top direction: distribution (how many neurons use it, sign)
proj=F@Vt[0]
print('loadings quantiles', np.quantile(proj,[0,.05,.25,.5,.75,.95,1]).round(2))
# Do L0 heads read this direction? 
for i in range(6):
    q,k,v,o=hw(0,i)
    print(i,'|q.x| %.2f |k.x| %.2f |v.x| %.2f'%(np.linalg.norm(q@Vt[0]),np.linalg.norm(k@Vt[0]),np.linalg.norm(v@Vt[0])))
print('ln1_b.0 cos with dir: %.2f; ln2_b.0 cos: %.2f'%(W['ln1_b.0']@Vt[0]/np.linalg.norm(W['ln1_b.0']), W['ln2_b.0']@Vt[0]/np.linalg.norm(W['ln2_b.0'])))
```

````