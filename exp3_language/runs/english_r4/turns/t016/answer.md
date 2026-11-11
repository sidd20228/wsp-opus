```python
svd=np.linalg.svd
def head(l,j):
    s=slice(j*64,(j+1)*64)
    g=W[f'ln1_w.{l}']
    Wq=W[f'W_q.{l}'][s]*g; Wk=W[f'W_k.{l}'][s]*g; Wv=W[f'W_v.{l}'][s]*g; Wo=W[f'W_o.{l}'][:,s]
    return Wq,Wk,Wv,Wo
H={(l,j):head(l,j) for l in range(4) for j in range(6)}
OV={k:v[3]@v[2] for k,v in H.items()}
rng=np.random.default_rng(0)
Qs=[np.linalg.qr(rng.standard_normal((384,384)))[0] for _ in range(30)]
def z(A,B):
    M=A@B; t=np.trace(M)/np.linalg.norm(M)
    null=[np.trace(A@Q@B@Q.T)/np.linalg.norm(A@Q@B@Q.T) for Q in Qs]
    return t,(t-np.mean(null))/np.std(null)
for l in range(1,4):
    for j in range(6):
        Wq,Wk,Wv,Wo=H[(l,j)]
        A=Wq.T@Wk
        out=[]
        for src in [(0,2),(0,3)]+[(1,i) for i in range(6)]+[(2,i) for i in range(6)]:
            if src[0]>=l: continue
            t,zz=z(A,OV[src])
            out.append(f"{src[0]}.{src[1]}:{zz:+.1f}")
        t,zs=z(A,np.eye(384))
        print(f"{l}.{j}", 'self z=%.1f'%((np.trace(A)/np.linalg.norm(A))/ (1/np.sqrt(384)) ),' '.join(out))
```