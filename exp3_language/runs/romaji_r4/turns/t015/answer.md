```python
import numpy as np
fro=np.linalg.norm
H=6;hd=64
def head(l,j):
    g=W[f'ln1_w.{l}'];s=slice(j*hd,(j+1)*hd)
    return W[f'W_q.{l}'][s]*g, W[f'W_k.{l}'][s]*g, W[f'W_v.{l}'][s]*g, W[f'W_o.{l}'][:,s]
Q3,K3,V3,O3=head(0,3);OV3=O3@V3
F=W['W_fc.0']*W['ln2_w.0']
so=np.linalg.svd(OV3,compute_uv=False)
vk={}
for j in [0,1,2,4,5]:
    Q,K,V,O=head(0,j);u,s,vt=np.linalg.svd(Q.T@K);vk[j]=vt[0]
    print('L0H%d key dir: |OV3 v| %.2f (OV3 sv1 %.2f, median-of-top20 %.2f) | MLP0 read |F v|/|F|_2 %.2f | each L0 head V read:'%(j,fro(OV3@vt[0]),so[0],np.median(so[:20]),fro(F@vt[0])/np.linalg.norm(F,2)),
          ' '.join('%.2f'%(fro(head(0,i)[2]@vt[0])/np.linalg.norm(head(0,i)[2],2)) for i in range(H)))
# random unit vector baseline
rng=np.random.default_rng(0);x=rng.normal(size=384);x/=fro(x)
print('random: |OV3 x| %.2f  |F x|/|F| %.2f'%(fro(OV3@x),fro(F@x)/np.linalg.norm(F,2)))
# query-side u of rank-1 heads: compare with ln1_b (constant query) and with r
for j in [1,2,4]:
    Q,K,V,O=head(0,j);b=W['ln1_b.0']
    qb=Q@b; u,s,vt=np.linalg.svd(Q.T@K)
    print('L0H%d: bias-query score on key dir v: %.2f ; typical u-score scale s1*|x|: %.2f'%(j,(qb@(K@vt[0])),s[0]*np.sqrt(384)))
```