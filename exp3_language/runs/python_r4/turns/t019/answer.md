```python
import numpy as np
hd=64
lw0=W['ln1_w.0']
def OVl(l,j):
    lw=W[f'ln1_w.{l}']
    return W[f'W_o.{l}'][:,j*hd:(j+1)*hd]@(W[f'W_v.{l}'][j*hd:(j+1)*hd]*lw)
k=24
_,_,Vt1=np.linalg.svd(OVl(0,1)); _,_,Vt5=np.linalg.svd(OVl(0,5))
Bt,_,_=np.linalg.svd(np.hstack([Vt1[:k].T,Vt5[:k].T]),full_matrices=False); B=Bt[:,:k]
def outsp(M,k):
    U,S,Vt=np.linalg.svd(M); return U[:,:k]
S1=outsp(OVl(0,1),20); S5=outsp(OVl(0,5),20); S4=outsp(OVl(0,4)+OVl(0,0),3)
L1=outsp(np.hstack([OVl(1,j) for j in range(6)]),60)
def frac(A,S): return (np.linalg.norm(A@S)**2/np.linalg.norm(A)**2)/(S.shape[1]/384)
print('subspace dims: B',B.shape[1],'S1',S1.shape[1],'S5',S5.shape[1],'S4',S4.shape[1],'L1',L1.shape[1])
for l in [1,2,3]:
    lw=W[f'ln1_w.{l}']
    for j in range(6):
        s=slice(j*hd,(j+1)*hd)
        r=[]
        for nm in ['q','k','v']:
            A=W[f'W_{nm}.{l}'][s]*lw
            r.append(nm+': B %.1f S1 %.1f S5 %.1f pos %.1f L1 %.1f'%(frac(A,B),frac(A,S1),frac(A,S5),frac(A,S4),frac(A,L1) if l>1 else 0))
        print(l,j,' || '.join(r))
```