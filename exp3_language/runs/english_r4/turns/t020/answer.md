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
T=svd(OV[(0,2)])[2][:20].T
Uu=svd(OV[(0,0)]+OV[(0,1)]+OV[(0,4)]+OV[(0,5)])[0][:,:3]
for l in range(1,4):
    for j in range(6):
        Wq,Wk,Wv,Wo=H[(l,j)]
        A=Wq.T@Wk; n2=np.linalg.norm(A)**2
        B=S.T@A@S
        fSS=np.linalg.norm(B)**2/n2/(20/384)**2
        anti=np.linalg.norm(B-B.T)/np.linalg.norm(B+B.T) if np.linalg.norm(B)>0 else 0
        sv=svd(B,compute_uv=False)[:4]
        fq=np.linalg.norm(Wq@S)**2/np.linalg.norm(Wq)**2/(20/384)
        fk=np.linalg.norm(Wk@S)**2/np.linalg.norm(Wk)**2/(20/384)
        print(f"{l}.{j} S-S enrich {fSS:5.1f} anti/sym {anti:.2f} Bsv {sv.round(2)} q-in-S {fq:.1f} k-in-S {fk:.1f}")
```