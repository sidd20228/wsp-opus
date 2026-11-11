```python
import numpy as np
def rows(m,j,l): 
    s=slice(j*64,(j+1)*64); return W[f'W_{m}.{l}'][s]
def colspace(M,k):
    u,s,vt=np.linalg.svd(M,full_matrices=False); return u[:,:k],s
# broad OV output directions in residual
OVs={}
for j in range(6):
    s=slice(j*64,(j+1)*64)
    OVs[j]=W['W_o.0'][:,s]@(W['W_v.0'][s]*W['ln1_w.0'])
Ub,sb=colspace(np.hstack([OVs[j] for j in [0,1,4,5]]),3)
print('broad OV out sv',sb[:8].round(2))
for l in [1,2,3]:
    D=W[f'ln1_w.{l}'];b=W[f'ln1_b.{l}']
    for j in range(6):
        Wq=rows('q',j,l);Wk=rows('k',j,l)
        qc=Wq@b
        kappa=(Wk*D).T@qc     # in normalized-x space: score += kappa . xhat_j
        # kappa acts on xhat (normalized residual); residual direction r contributes kappa.r (scaled)
        fr=np.linalg.norm(Ub.T@kappa)/np.linalg.norm(kappa)
        print(f'L{l}H{j} |kappa|={np.linalg.norm(kappa):.2f} frac in broad-out top3={fr:.2f}  per dir {(Ub.T@kappa).round(2)}')
```