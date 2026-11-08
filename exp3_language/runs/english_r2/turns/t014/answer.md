```python
def head(l,j):
    s=slice(64*j,64*j+64)
    w=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][s]*w, W[f'W_k.{l}'][s]*w, W[f'W_v.{l}'][s]*w, W[f'W_o.{l}'][:,s]
QK={};OV={}
for l in range(4):
    for j in range(6):
        Wq,Wk,Wv,Wo=head(l,j); QK[l,j]=Wq.T@Wk; OV[l,j]=Wo@Wv
u,s,vt=np.linalg.svd(QK[0,2])
k=20
B,_=np.linalg.qr(np.hstack([u[:,:k],vt[:k].T]))
r=np.linalg.matrix_rank(B); print('dim P',B.shape)
# principal angles between query and key subspaces
print('cos principal angles q vs k', np.linalg.svd(u[:,:k].T@vt[:k].T,compute_uv=False).round(2))
def frac(M,Bs):
    return (np.linalg.norm(Bs.T@M)**2/np.linalg.norm(M)**2, np.linalg.norm(M@Bs)**2/np.linalg.norm(M)**2)
print('baseline', B.shape[1]/384)
for l in range(4):
    for j in range(6):
        fq,fk=frac(QK[l,j],B)
        fPP=np.linalg.norm(B.T@QK[l,j]@B)**2/np.linalg.norm(QK[l,j])**2
        fvin=np.linalg.norm(OV[l,j]@B)**2/np.linalg.norm(OV[l,j])**2
        print(f'L{l}H{j} QK q-side {fq:.2f} k-side {fk:.2f} PxP {fPP:.2f} | OV input in P {fvin:.2f}')
```