```python
def head(l,j):
    s=slice(64*j,64*j+64)
    w=W[f'ln1_w.{l}']
    Wq=W[f'W_q.{l}'][s]*w; Wk=W[f'W_k.{l}'][s]*w; Wv=W[f'W_v.{l}'][s]*w
    Wo=W[f'W_o.{l}'][:,s]
    return Wq,Wk,Wv,Wo
QK={};OV={}
for l in range(4):
    for j in range(6):
        Wq,Wk,Wv,Wo=head(l,j)
        QK[l,j]=Wq.T@Wk; OV[l,j]=Wo@Wv
for l in range(4):
    for j in range(6):
        print(f'L{l}H{j} QKfro {np.linalg.norm(QK[l,j]):7.2f} QKsv1 {np.linalg.svd(QK[l,j],compute_uv=False)[0]:6.2f} OVfro {np.linalg.norm(OV[l,j]):6.2f} OVsv1 {np.linalg.svd(OV[l,j],compute_uv=False)[0]:6.2f}')
```