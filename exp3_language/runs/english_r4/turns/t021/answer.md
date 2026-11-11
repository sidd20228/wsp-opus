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
g0=W['ln1_w.0']
for j in [0,1,4,5,3,2]:
    ov=OV[(0,j)]
    U,s,Vt=svd(ov)
    # Note OV acts on z (normalized, unscaled); output to residual x. Input direction in residual coords ~ same basis.
    for r in range(3):
        cos=U[:,r]@Vt[r]
        print(j,r,'sv',round(s[r],2),'cos(out,in)',round(cos,2),'in in S',round(np.linalg.norm(S.T@Vt[r]),2),'out in S',round(np.linalg.norm(S.T@U[:,r]),2))
    B=S.T@ov@S
    print('   S-restricted OV eig',np.sort_complex(np.linalg.eigvals(B))[:3].round(2),np.sort_complex(np.linalg.eigvals(B))[-3:].round(2))
```