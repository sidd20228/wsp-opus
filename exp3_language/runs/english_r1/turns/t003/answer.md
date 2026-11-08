<invoke name="python">
```python
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=200)
H=6; hd=64
def heads(l):
    w=W[f'ln1_w.{l}']
    res=[]
    for j in range(H):
        s=slice(j*hd,(j+1)*hd)
        Wq=W[f'W_q.{l}'][s]*w; Wk=W[f'W_k.{l}'][s]*w; Wv=W[f'W_v.{l}'][s]*w; Wo=W[f'W_o.{l}'][:,s]
        res.append(dict(Wq=Wq,Wk=Wk,Wv=Wv,Wo=Wo,QK=Wq.T@Wk,OV=Wo@Wv))
    return res
Hs=[heads(l) for l in range(4)]
for l in range(4):
    print('layer',l)
    for j in range(H):
        h=Hs[l][j]
        QK=h['QK']; OV=h['OV']
        sym=(QK+QK.T)/2; asym=(QK-QK.T)/2
        ev=np.linalg.eigvals(OV); 
        print(j,'|QK|F %.2f |OV|F %.2f  asym/sym %.2f  QKtrace %.2f  OVtrace %.2f  OVeig posfrac %.2f'%(np.linalg.norm(QK),np.linalg.norm(OV),np.linalg.norm(asym)/np.linalg.norm(sym),np.trace(QK),np.trace(OV), (ev.real*np.abs(ev)).sum()/np.abs(ev**2).sum()),
              'qbias %.2f kbias %.2f'%(np.linalg.norm(W[f'W_q.{l}'][j*hd:(j+1)*hd]@W[f'ln1_b.{l}']), np.linalg.norm(W[f'W_k.{l}'][j*hd:(j+1)*hd]@W[f'ln1_b.{l}'])))
```

````