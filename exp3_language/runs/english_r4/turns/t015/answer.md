```python
svd=np.linalg.svd
def head(l,j):
    s=slice(j*64,(j+1)*64)
    g=W[f'ln1_w.{l}']
    Wq=W[f'W_q.{l}'][s]*g; Wk=W[f'W_k.{l}'][s]*g; Wv=W[f'W_v.{l}'][s]*g; Wo=W[f'W_o.{l}'][:,s]
    return Wq,Wk,Wv,Wo
H={(l,j):head(l,j) for l in range(4) for j in range(6)}
OV={k:v[3]@v[2] for k,v in H.items()}
for src in [(0,2),(0,3),(0,0)]:
    print('source',src)
    for l in range(1,4):
        row=[]
        for j in range(6):
            Wq,Wk,Wv,Wo=H[(l,j)]
            M=Wq.T@Wk@OV[src]
            row.append(f"{np.trace(M)/np.linalg.norm(M):+.2f}")
        print(' layer',l,row)
# also self-matching: trace of QK itself (same-token attention)
for l in range(4):
    print('self tr/F',l,[f"{np.trace(H[(l,j)][0].T@H[(l,j)][1])/np.linalg.norm(H[(l,j)][0].T@H[(l,j)][1]):+.2f}" for j in range(6)])
# OV trace (copying in residual basis)
for l in range(4):
    print('OV tr/F',l,[f"{np.trace(OV[(l,j)])/np.linalg.norm(OV[(l,j)]):+.2f}" for j in range(6)])
```