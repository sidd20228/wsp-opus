```python
def head(l,j):
    s=slice(j*64,(j+1)*64)
    g=W[f'ln1_w.{l}']
    Wq=W[f'W_q.{l}'][s]*g; Wk=W[f'W_k.{l}'][s]*g; Wv=W[f'W_v.{l}'][s]*g; Wo=W[f'W_o.{l}'][:,s]
    return Wq,Wk,Wv,Wo
H={(l,j):head(l,j) for l in range(4) for j in range(6)}
OV={k:v[3]@v[2] for k,v in H.items()}
svd=np.linalg.svd
Wq,Wk,Wv,Wo=H[(0,2)]
U,s,Vt=svd(Wq.T@Wk)
# subspaces in *raw residual* coords: undo g scaling? Use raw: directions in z-space; convert to x-space approx by multiplying g
g=W['ln1_w.0']
# Stack all layer-0 read matrices (in z coords) and look at the rank
R=np.vstack([W['W_q.0']*g,W['W_k.0']*g,W['W_v.0']*g])
print('layer0 attn read sv',svd(R,compute_uv=False)[:80].round(1))
F=W['W_fc.0']*W['ln2_w.0']
print('fc0 read sv',svd(F,compute_uv=False)[:80].round(1))
print('raw tok-subspace guess: sv of stacked V of all layer0 heads',svd(W['W_v.0']*g,compute_uv=False)[:60].round(2))
```