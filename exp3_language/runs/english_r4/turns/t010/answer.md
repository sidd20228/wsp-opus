```python
def head(l,j):
    s=slice(j*64,(j+1)*64)
    g=W[f'ln1_w.{l}']
    Wq=W[f'W_q.{l}'][s]*g; Wk=W[f'W_k.{l}'][s]*g; Wv=W[f'W_v.{l}'][s]*g; Wo=W[f'W_o.{l}'][:,s]
    return Wq,Wk,Wv,Wo
H={(l,j):head(l,j) for l in range(4) for j in range(6)}
OV={k:v[3]@v[2] for k,v in H.items()}
def svd(M): return np.linalg.svd(M)
Wq,Wk,Wv,Wo=H[(0,2)]
U,s,Vt=svd(Wq.T@Wk)
S=Vt[:18].T  # key-side subspace
Sq=U[:,:18]
_,_,Vt2=svd(OV[(0,2)]); T=Vt2[:20].T
_,_,Vtu=svd(OV[(0,0)]); Vu=Vtu[:3].T
for j in [0,1,3,4,5]:
    Wq,Wk,Wv,Wo=H[(0,j)]
    U,s,Vt=svd(Wq.T@Wk)
    for r in range(2):
        u=U[:,r]; v=Vt[r]
        print(j,r,round(s[r],2),'q in S',round(np.linalg.norm(S.T@u),2),'q in T',round(np.linalg.norm(T.T@u),2),'q in Vu',round(np.linalg.norm(Vu.T@u),2),
              '| k in S',round(np.linalg.norm(S.T@v),2),'k in T',round(np.linalg.norm(T.T@v),2),'k in Vu',round(np.linalg.norm(Vu.T@v),2))
# restricted eigenvalues of 0.2
Wq,Wk,Wv,Wo=H[(0,2)]
A=Wq.T@Wk
P=np.linalg.qr(np.hstack([S,Sq]))[0][:,:24]
B=P.T@A@P
ev=np.linalg.eigvals(B)
ev=ev[np.argsort(-np.abs(ev))]
print('eig |.|',np.abs(ev).round(2))
print('angle deg',np.degrees(np.angle(ev)).round(1))
```