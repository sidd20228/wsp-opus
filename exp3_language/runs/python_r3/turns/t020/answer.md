```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
w0=W['ln1_w.0']; H=w0>0.45
def hq(l,j):
    s=slice(64*j,64*j+64); wl=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][s]*wl, W[f'W_k.{l}'][s]*wl, W[f'W_v.{l}'][s]*wl, W[f'W_o.{l}'][:,s]
rows=np.concatenate([W['W_v.0'][64*j:64*j+64] for j in [1,2,3,5]],0)
T=np.linalg.svd(rows,full_matrices=False)[2][:27].T
Hb=np.eye(384)[:,H]
def outsp(l,i,k):
    q,kk,v,o=hq(l,i); U,S,Vt=np.linalg.svd(o@v); return U[:,:k], S[:k]
src={'T':T,'Hpe':Hb}
for i,kd in [(0,2),(1,27),(2,10),(3,10),(4,3),(5,27)]:
    src[f'O0.{i}'],S=outsp(0,i,kd)
U,S,Vt=np.linalg.svd(W['W_proj.0']); src['MLP0']=U[:,:100]
print({k:v.shape[1] for k,v in src.items()})
names=list(src)
for L in [1,2]:
    for h in range(6):
        q,k,v,o=hq(L,h)
        print(f'L{L}H{h} q-read:', ' '.join(f"{n}={np.linalg.norm(q@src[n])/np.linalg.norm(q):.2f}" for n in names))
        print(f'      k-read:', ' '.join(f"{n}={np.linalg.norm(k@src[n])/np.linalg.norm(k):.2f}" for n in names))
        print(f'      v-read:', ' '.join(f"{n}={np.linalg.norm(v@src[n])/np.linalg.norm(v):.2f}" for n in names))
# random baseline: fraction expected sqrt(dim/384)
print({n:round(float(np.sqrt(src[n].shape[1]/384)),2) for n in names})
```