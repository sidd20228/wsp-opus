```python
import numpy as np
def rows(m,j,l): 
    s=slice(j*64,(j+1)*64); return W[f'W_{m}.{l}'][s]*W[f'ln1_w.{l}']
def OV(l,j):
    s=slice(j*64,(j+1)*64); return W[f'W_o.{l}'][:,s]@rows('v',j,l)
M=np.vstack([rows('q',2,0),rows('k',2,0)])
_,_,vt=np.linalg.svd(M,full_matrices=False); P=vt[:24].T
OVb=np.hstack([OV(0,j) for j in [0,1,4,5]]); ub,_,_=np.linalg.svd(OVb,full_matrices=False); Bo=ub[:,:3]
u2,_,_=np.linalg.svd(OV(0,2)); O2=u2[:,:20]
u3,_,_=np.linalg.svd(OV(0,3)); O3=u3[:,:5]
spaces={'P':P,'Bo':Bo,'O2':O2,'O3':O3}
for l in [1,2]:
  for j in range(6):
    Wq=rows('q',j,l);Wk=rows('k',j,l)
    # SVD of QK
    U,S,Vt=np.linalg.svd(Wq.T@Wk)
    print(f'L{l}H{j} sv',S[:6].round(1))
    for i in range(4):
        qd=U[:,i];kd=Vt[i]
        print('   ',i,'q:',' '.join(f'{n}={np.linalg.norm(X.T@qd):.2f}' for n,X in spaces.items()),'| k:',' '.join(f'{n}={np.linalg.norm(X.T@kd):.2f}' for n,X in spaces.items()))
```