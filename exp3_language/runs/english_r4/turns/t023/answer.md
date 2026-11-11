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
U2=svd(OV[(0,2)])[0][:,:24]
Uu=svd(OV[(0,0)]+OV[(0,1)]+OV[(0,4)]+OV[(0,5)])[0][:,:3]
def tr_in(A,B): return np.trace(B.T@A@B)
for l in range(1,4):
    for j in range(6):
        Wq,Wk,Wv,Wo=H[(l,j)]
        A=Wq.T@Wk; nA=np.linalg.norm(A)
        print(f"{l}.{j} trace total {np.trace(A)/nA:+.2f} | in S {tr_in(A,S)/nA:+.2f} in T {tr_in(A,T)/nA:+.2f} in O2 {tr_in(A,U2)/nA:+.2f} | T->O2 (q tok,k prevtok) {np.trace(T.T@A@OV[(0,2)]@T)/np.linalg.norm(A@OV[(0,2)]):+.2f} | blocks T-T {np.linalg.norm(T.T@A@T)/nA:.2f} T-O2 {np.linalg.norm(T.T@A@U2)/nA:.2f} O2-T {np.linalg.norm(U2.T@A@T)/nA:.2f} O2-O2 {np.linalg.norm(U2.T@A@U2)/nA:.2f} rnd~{24/384:.2f}")
```