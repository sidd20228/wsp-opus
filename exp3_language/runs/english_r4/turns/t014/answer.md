```python
svd=np.linalg.svd
def head(l,j):
    s=slice(j*64,(j+1)*64)
    g=W[f'ln1_w.{l}']
    Wq=W[f'W_q.{l}'][s]*g; Wk=W[f'W_k.{l}'][s]*g; Wv=W[f'W_v.{l}'][s]*g; Wo=W[f'W_o.{l}'][:,s]
    return Wq,Wk,Wv,Wo
H={(l,j):head(l,j) for l in range(4) for j in range(6)}
OV={k:v[3]@v[2] for k,v in H.items()}
Uu_,su,Vtu=svd(OV[(0,0)]+OV[(0,1)]+OV[(0,4)]+OV[(0,5)]); Uu=Uu_[:,:3]
for l in range(1,4):
    b=W[f'ln1_b.{l}']
    for j in range(6):
        s=slice(j*64,(j+1)*64)
        Wq,Wk,Wv,Wo=H[(l,j)]
        qb=W[f'W_q.{l}'][s]@b; kb=W[f'W_k.{l}'][s]@b
        kdir=Wk.T@qb
        # typical query norm for unit-variance z: ||Wq||_F
        print(l,j,'|qb|',round(np.linalg.norm(qb),2),'typ |q|',round(np.linalg.norm(Wq),2),'kdir norm',round(np.linalg.norm(kdir),2),'frac kdir in Uu',round(np.linalg.norm(Uu.T@kdir)**2/np.linalg.norm(kdir)**2,3),
              'kdir.Uu',(Uu.T@kdir).round(2))
```