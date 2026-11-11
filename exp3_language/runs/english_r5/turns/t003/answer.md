```python
import numpy as np
hd=64
def heads(l):
    D=W[f'ln1_w.{l}']
    res=[]
    for j in range(6):
        s=slice(j*64,(j+1)*64)
        Wq=W[f'W_q.{l}'][s]*D; Wk=W[f'W_k.{l}'][s]*D; Wv=W[f'W_v.{l}'][s]*D; Wo=W[f'W_o.{l}'][:,s]
        res.append(dict(Wq=Wq,Wk=Wk,Wv=Wv,Wo=Wo,QK=Wq.T@Wk,OV=Wo@Wv))
    return res
H=[heads(l) for l in range(4)]
# norms and basic
for l in range(4):
    for j in range(6):
        h=H[l][j]
        QK=h['QK'];OV=h['OV']
        S=(QK+QK.T)/2;A=(QK-QK.T)/2
        ev=np.linalg.eigvals(h['Wk']@h['Wq'].T)  # nonzero eigs of QK
        evo=np.linalg.eigvals(h['Wv']@h['Wo'])
        print(f'L{l}H{j} |QK|={np.linalg.norm(QK):.1f} |OV|={np.linalg.norm(OV):.2f} sym/anti={np.linalg.norm(S)/np.linalg.norm(A):.2f} QKtrpos={ev.real.sum()/np.abs(ev).sum():.2f} OVpos={evo.real.sum()/np.abs(evo).sum():.2f} QKsv={np.round(np.linalg.svd(QK,compute_uv=False)[:5],1)}')
```