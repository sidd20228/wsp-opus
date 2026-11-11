```python
import numpy as np
def rows(m,j,l): 
    s=slice(j*64,(j+1)*64); return W[f'W_{m}.{l}'][s]*W[f'ln1_w.{l}']
def OV(l,j):
    s=slice(j*64,(j+1)*64); return W[f'W_o.{l}'][:,s]@rows('v',j,l)
def QK(l,j): return rows('q',j,l).T@rows('k',j,l)
OVb=np.hstack([OV(0,j) for j in [0,1,4,5]]); ub,sb,_=np.linalg.svd(OVb,full_matrices=False); Bo=ub[:,:3]
print('Bo sv',sb[:4].round(2))
np.set_printoptions(suppress=True)
for l in [1,2,3]:
    for j in range(6):
        A=QK(l,j)
        b=Bo.T@A@Bo
        b_=W[f'ln1_b.{l}']
        Wq=W[f'W_q.{l}'][j*64:(j+1)*64];Wk=W[f'W_k.{l}'][j*64:(j+1)*64]
        kap=(Wk*W[f'ln1_w.{l}']).T@(Wq@b_)
        print(f'L{l}H{j} BoBo=\n',b.round(2),' eig',np.linalg.eigvals(b).round(2),' kappa.Bo',(Bo.T@kap).round(2), '|QK|',np.linalg.norm(A).round(1))
```