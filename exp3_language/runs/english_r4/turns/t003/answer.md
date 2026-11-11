```python
def head(l,j):
    s=slice(j*64,(j+1)*64)
    g=W[f'ln1_w.{l}']
    Wq=W[f'W_q.{l}'][s]*g; Wk=W[f'W_k.{l}'][s]*g; Wv=W[f'W_v.{l}'][s]*g; Wo=W[f'W_o.{l}'][:,s]
    return Wq,Wk,Wv,Wo
for l in range(4):
    for j in range(6):
        Wq,Wk,Wv,Wo=head(l,j)
        qk=Wq.T@Wk; ov=Wo@Wv
        sq=np.linalg.svd(qk,compute_uv=False); so=np.linalg.svd(ov,compute_uv=False)
        print(l,j,'QK',sq[:8].round(2),'OV',so[:6].round(2))
```