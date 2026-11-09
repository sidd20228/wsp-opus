```python
import numpy as np
hd=64
def OVl(l,j):
    lw_=W[f'ln1_w.{l}']
    return W[f'W_o.{l}'][:,j*hd:(j+1)*hd]@(W[f'W_v.{l}'][j*hd:(j+1)*hd]*lw_)
E=np.load('/tmp/E40.npy')
k=E.shape[1]
np.set_printoptions(linewidth=250,suppress=True)
Ds=[]
for (l,j) in [(3,5),(3,1),(2,1),(3,0)]:
    D=E.T@OVl(l,j)@E
    Ds.append(D/np.abs(D).mean())
C=np.corrcoef(np.array([d.ravel() for d in Ds]))
print('corr between heads D matrices',np.round(C,2))
D=Ds[0]+Ds[1]
print('diag',np.round(np.diag(D),1))
for i in range(k):
    print('%2d '%i+' '.join('%5.1f'%x for x in D[i]))
```