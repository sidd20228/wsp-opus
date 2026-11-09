```python
import numpy as np
hd=64
lw0=W['ln1_w.0']
def OV0(j): return W['W_o.0'][:,j*hd:(j+1)*hd]@(W['W_v.0'][j*hd:(j+1)*hd]*lw0)
k=24
_,_,Vt1=np.linalg.svd(OV0(1)); B1=Vt1[:k].T
_,_,Vt5=np.linalg.svd(OV0(5)); B5=Vt5[:k].T
# token basis: union
Bt,_,_=np.linalg.svd(np.hstack([B1,B5]),full_matrices=False); B=Bt[:,:k]
print('overlap B1,B5', np.round(np.linalg.svd(B1.T@B5,compute_uv=False),2))
rng=np.random.default_rng(1)
def eigratio(M):
    ev=np.linalg.eigvals(M); return ev.real.sum()/np.abs(ev).sum()
for l in [1,2,3]:
    lw=W[f'ln1_w.{l}']
    for j in range(6):
        s=slice(j*hd,(j+1)*hd)
        QK=(W[f'W_q.{l}'][s]*lw).T@(W[f'W_k.{l}'][s]*lw)
        out=[]
        for prev in [1,5]:
            Mi=B.T@QK@OV0(prev)@B
            out.append('ind(H%d) r=%.2f n=%.2f'%(prev,eigratio(Mi),np.linalg.norm(Mi)))
        Ms=B.T@QK@B
        out.append('same r=%.2f n=%.2f'%(eigratio(Ms),np.linalg.norm(Ms)))
        print(l,j,' | '.join(out))
```