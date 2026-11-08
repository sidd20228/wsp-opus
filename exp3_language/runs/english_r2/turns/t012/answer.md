```python
def head(l,j):
    s=slice(64*j,64*j+64)
    w=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][s]*w, W[f'W_k.{l}'][s]*w, W[f'W_v.{l}'][s]*w, W[f'W_o.{l}'][:,s]
QK={};OV={}
for l in range(4):
    for j in range(6):
        Wq,Wk,Wv,Wo=head(l,j); QK[l,j]=Wq.T@Wk; OV[l,j]=Wo@Wv
S=[222,78,360,103,72,254,25,197,273,136,307,249,292,226,302,130]
mask=np.ones(384); mask[S]=0
D=np.diag(mask)
def nf(A): return np.linalg.norm(A)
QKm={k:D@v@D for k,v in QK.items()}; OVm={k:D@v@D for k,v in OV.items()}
print('baseline 1/sqrt(368)=%.3f'%(1/np.sqrt(368)))
for l in range(1,4):
    print('Layer',l)
    for j in range(6):
        rowK=[nf(QKm[l,j]@OVm[m,i])/(nf(QKm[l,j])*nf(OVm[m,i])) for m in range(l) for i in range(6)]
        rowQ=[nf(OVm[m,i].T@QKm[l,j])/(nf(QKm[l,j])*nf(OVm[m,i])) for m in range(l) for i in range(6)]
        rowV=[nf(OVm[l,j]@OVm[m,i])/(nf(OVm[l,j])*nf(OVm[m,i])) for m in range(l) for i in range(6)]
        print(f' L{l}H{j} K:', ' '.join(f'{x:.2f}' for x in rowK))
        print(f'       Q:', ' '.join(f'{x:.2f}' for x in rowQ))
        print(f'       V:', ' '.join(f'{x:.2f}' for x in rowV))
```