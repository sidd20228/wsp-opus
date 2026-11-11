```python
svd=np.linalg.svd
def head(l,j):
    s=slice(j*64,(j+1)*64)
    g=W[f'ln1_w.{l}']
    Wq=W[f'W_q.{l}'][s]*g; Wk=W[f'W_k.{l}'][s]*g; Wv=W[f'W_v.{l}'][s]*g; Wo=W[f'W_o.{l}'][:,s]
    return Wq,Wk,Wv,Wo
H={(l,j):head(l,j) for l in range(4) for j in range(6)}
OV={k:v[3]@v[2] for k,v in H.items()}
U,s,Vt=svd(H[(0,2)][0].T@H[(0,2)][1]); S=np.linalg.qr(np.hstack([Vt[:18].T,U[:,:18]]))[0][:,:20]
T=svd(OV[(0,2)])[2][:24].T
O2=svd(OV[(0,2)])[0][:,:24]
Uu=svd(OV[(0,0)]+OV[(0,1)]+OV[(0,4)]+OV[(0,5)])[0][:,:3]
M0=svd(W['W_proj.0'])[0][:,:40]
O3=svd(OV[(0,3)])[0][:,:10]
b1=W['ln1_b.1']; bdir=b1/np.linalg.norm(b1)
subs={'S':S,'T':T,'O2':O2,'Uu':Uu,'M0':M0,'O3':O3}
for l in [1,2]:
    for j in range(6):
        Wq,Wk,Wv,Wo=H[(l,j)]
        U,s,Vt=svd(Wq.T@Wk)
        for r in range(3):
            q=U[:,r]; k=Vt[r]
            qs=' '.join(f"{n}:{np.linalg.norm(B.T@q)**2:.2f}" for n,B in subs.items())
            ks=' '.join(f"{n}:{np.linalg.norm(B.T@k)**2:.2f}" for n,B in subs.items())
            print(f"{l}.{j} r{r} sv {s[r]:.2f} | q {qs} | k {ks}")
```