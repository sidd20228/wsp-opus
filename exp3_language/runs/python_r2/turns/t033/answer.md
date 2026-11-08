```python
import numpy as np
hd=64
sl=lambda h: slice(h*hd,(h+1)*hd)
D0=W['ln1_w.0']; D2=W['ln2_w.0']
def rsvd(M):
    U,S,Vt=np.linalg.svd(M,full_matrices=False); return Vt,S
Vqk,_=rsvd(np.vstack([W['W_q.0'][sl(1)],W['W_k.0'][sl(1)],W['W_q.0'][sl(5)],W['W_k.0'][sl(5)]]))
P20=Vqk[:20].T
V1,s1=rsvd(W['W_v.0'][sl(1)])
# x-hat functionals
F=W['W_fc.0']*D2
alive=np.linalg.norm(F,axis=1)>0.6
Vm,sm=rsvd(F[alive])
M60=Vm[:60].T; M30=Vm[:30].T
print('dir sv  posQK20  MLP0top30  MLP0top60')
for i in range(32):
    v=V1[i]; fx=D0*v; fx/=np.linalg.norm(fx)
    print(i,'%.2f'%s1[i],'%.2f'%np.linalg.norm(P20.T@v),'%.2f'%np.linalg.norm(M30.T@fx),'%.2f'%np.linalg.norm(M60.T@fx))
```