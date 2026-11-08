```python
def head(l,j):
    s=slice(64*j,64*j+64)
    ww=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][s]*ww, W[f'W_k.{l}'][s]*ww, W[f'W_v.{l}'][s]*ww, W[f'W_o.{l}'][:,s]
a,b,c,d=head(0,2)
R=W['W_fc.0']*W['ln2_w.0']
nr=np.linalg.norm(R,axis=1); Ra=R[nr>1.0]
print('active',Ra.shape)
_,sr,vr=np.linalg.svd(Ra,full_matrices=False)
print('Ra sv',sr[:100].round(2))
_,sv_,vv=np.linalg.svd(c,full_matrices=False)
for kr in [40,60,80,100,150]:
    cosang=np.linalg.svd(vv[:64]@vr[:kr].T,compute_uv=False)
    print(kr, 'cos principal angles WvH2(64) vs MLP0 top',kr, cosang[:40].round(2))
```