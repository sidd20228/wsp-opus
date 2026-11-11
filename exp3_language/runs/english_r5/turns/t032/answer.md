```python
import numpy as np
def rows(m,j,l,fold=True): 
    s=slice(j*64,(j+1)*64); return W[f'W_{m}.{l}'][s]*(W[f'ln1_w.{l}'] if fold else 1)
s=slice(128,192)
_,_,vtr=np.linalg.svd(W['W_v.0'][s],full_matrices=False)
Tr=vtr[:20].T
O2=W['W_o.0'][:,s]@(W['W_v.0'][s]*W['ln1_w.0'])
uo,so,_=np.linalg.svd(O2); O2s=uo[:,:20]
Wo2=W['W_o.0'][:,s]; uw,sw,_=np.linalg.svd(Wo2); O2w=uw[:,:20]
print('overlap Tr vs O2 out',np.linalg.svd(Tr.T@O2s,compute_uv=False).round(2)[:8])
print('overlap Tr vs Wo2 out',np.linalg.svd(Tr.T@O2w,compute_uv=False).round(2)[:8])
def frac(R,X): return np.linalg.norm(R@X)**2/np.linalg.norm(R)**2
print('baseline',20/384)
for l in [1,2,3]:
    for j in range(6):
        print(f'L{l}H{j}', ' '.join(f'{m}: Tr={frac(rows(m,j,l),Tr):.3f} O2={frac(rows(m,j,l),O2w):.3f}' for m in 'qkv'))
```