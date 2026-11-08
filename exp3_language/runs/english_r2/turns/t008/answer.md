```python
def head(l,j):
    s=slice(64*j,64*j+64)
    w=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][s]*w, W[f'W_k.{l}'][s]*w, W[f'W_v.{l}'][s]*w, W[f'W_o.{l}'][:,s]
QK={};OV={}
for l in range(4):
    for j in range(6):
        Wq,Wk,Wv,Wo=head(l,j); QK[l,j]=Wq.T@Wk; OV[l,j]=Wo@Wv
def nf(A): return np.linalg.norm(A)
# K composition: QK_l @ OV_m (key side reads OV_m output); Q comp: OV_m^T @ QK_l
for l in range(1,4):
    print('Layer',l)
    for j in range(6):
        row=[]
        for m in range(l):
            for i in range(6):
                k=nf(QK[l,j]@OV[m,i])/(nf(QK[l,j])*nf(OV[m,i]))
                row.append(k)
        print(f' L{l}H{j} K:', ' '.join(f'{x:.2f}' for x in row))
    for j in range(6):
        row=[]
        for m in range(l):
            for i in range(6):
                k=nf(OV[m,i].T@QK[l,j])/(nf(QK[l,j])*nf(OV[m,i]))
                row.append(k)
        print(f' L{l}H{j} Q:', ' '.join(f'{x:.2f}' for x in row))
```