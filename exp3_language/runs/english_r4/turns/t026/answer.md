```python
svd=np.linalg.svd
def head(l,j):
    s=slice(j*64,(j+1)*64)
    g=W[f'ln1_w.{l}']
    Wq=W[f'W_q.{l}'][s]*g; Wk=W[f'W_k.{l}'][s]*g; Wv=W[f'W_v.{l}'][s]*g; Wo=W[f'W_o.{l}'][:,s]
    return Wq,Wk,Wv,Wo
H={(l,j):head(l,j) for l in range(4) for j in range(6)}
OV={k:v[3]@v[2] for k,v in H.items()}
# use raw (unscaled by g) input directions: multiply back to residual coordinates: reading vector in x-space ~ W*g (already). fine.
V02=svd(OV[(0,2)])[2]
V31=svd(OV[(3,1)])[2]
V35=svd(OV[(3,5)])[2]
for k in [24,28,32,40]:
    print(k,'0.2 vs 3.1',svd(V02[:k]@V31[:k].T,compute_uv=False).round(2))
for k in [28,40]:
    print(k,'0.2 vs 3.5',svd(V02[:k]@V35[:k].T,compute_uv=False).round(2))
    print(k,'3.1 vs 3.5',svd(V31[:k]@V35[:k].T,compute_uv=False).round(2))
```