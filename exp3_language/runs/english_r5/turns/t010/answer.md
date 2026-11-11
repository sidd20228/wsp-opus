```python
import numpy as np
def rows(m,j,l=0): 
    s=slice(j*64,(j+1)*64); return W[f'W_{m}.{l}'][s]*W[f'ln1_w.{l}']
Wq=rows('q',2);Wk=rows('k',2)
B=Wq.T@Wk
ev,evec=np.linalg.eig(B)
o=np.argsort(-np.abs(ev))[:24]
ev=ev[o];evec=evec[:,o]
Vb=np.vstack([rows('v',j) for j in [0,1,4,5]])
u,s,vt=np.linalg.svd(Vb,full_matrices=False)
top=vt[:6]
# orthonormalize eigvec planes
for i in range(0,24):
    e=evec[:,i]
    P=np.linalg.qr(np.stack([e.real,e.imag],1))[0] if abs(e.imag).sum()>1e-6 else (e.real/np.linalg.norm(e.real))[:,None]
    proj=np.linalg.norm(top@P,axis=1)
    print(f'{abs(ev[i]):.2f} {np.degrees(np.angle(ev[i])):7.1f}', proj.round(2))
```