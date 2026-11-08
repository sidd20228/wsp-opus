<invoke name="python">
```python
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=200)
H=6; hd=64
w=W['ln1_w.0']; b=W['ln1_b.0']
U=[];V=[]
for j in range(H):
    s=slice(j*hd,(j+1)*hd)
    Wq=W['W_q.0'][s]*w; Wk=W['W_k.0'][s]*w
    QK=Wq.T@Wk
    u,sv,vt=np.linalg.svd(QK)
    U.append(u[:,:2]); V.append(vt[:2].T)
Ufc,sfc,Vfc=np.linalg.svd(W['W_fc.0']*w)
vfc=Vfc[:2].T
names=['H%d'%j for j in range(6)]
print('|cos| between top query dirs u1:')
print(np.abs(np.array([[U[a][:,0]@U[c][:,0] for c in range(6)] for a in range(6)])))
print('|cos| between top key dirs v1:')
print(np.abs(np.array([[V[a][:,0]@V[c][:,0] for c in range(6)] for a in range(6)])))
print('u1 vs v1 same head', [round(float(U[a][:,0]@V[a][:,0]),3) for a in range(6)])
print('fc top dir vs u1', [round(float(abs(vfc[:,0]@U[a][:,0])),3) for a in range(6)], 'vs v1',[round(float(abs(vfc[:,0]@V[a][:,0])),3) for a in range(6)])
# ln1_b.0 direction vs
bn=b*0+b; 
print('ln1_w.0 mask large frac in u1:', [round(float((U[a][:,0]**2)[w>0.3].sum()),3) for a in range(6)])
print('sfc', sfc[:5])
```

````