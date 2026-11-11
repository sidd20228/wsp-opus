```python
import numpy as np
def rows(m,j,l,fold=True): 
    s=slice(j*64,(j+1)*64); return W[f'W_{m}.{l}'][s]*(W[f'ln1_w.{l}'] if fold else 1)
s=slice(128,192)
_,sv,vt=np.linalg.svd(W['W_v.0'][s]*W['ln1_w.0'],full_matrices=False)
T=vt[:20].T   # token subspace guess in xhat space (as read with D)
_,_,vtr=np.linalg.svd(W['W_v.0'][s],full_matrices=False)
Tr=vtr[:20].T
M=np.vstack([rows('q',2,0),rows('k',2,0)]); _,_,vp=np.linalg.svd(M,full_matrices=False); P=vp[:20].T
print('T vs P overlap',np.linalg.svd(T.T@P,compute_uv=False).round(2)[:6])
print('Tr vs P overlap',np.linalg.svd(Tr.T@P,compute_uv=False).round(2)[:6])
def frac(R,X): return np.linalg.norm(R@X)**2/np.linalg.norm(R)**2
base=20/384
print('baseline',round(base,3))
for name,R in [('fc0',W['W_fc.0']*W['ln2_w.0']),('fc0raw',W['W_fc.0'])]+[(f'v1{j}',rows('v',j,1)) for j in range(6)]+[(f'q1{j}',rows('q',j,1)) for j in range(6)]+[(f'k1{j}',rows('k',j,1)) for j in range(6)]+[(f'v3{j}',rows('v',j,3)) for j in range(6)]+[(f'q3{j}',rows('q',j,3)) for j in range(6)]:
    print(name,'T',round(frac(R,T),3),'Tr',round(frac(R,Tr),3),'P',round(frac(R,P),3))
```