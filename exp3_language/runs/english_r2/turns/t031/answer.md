```python
from sklearn.decomposition import PCA
def head(l,j):
    s=slice(64*j,64*j+64)
    ww=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][s]*ww, W[f'W_k.{l}'][s]*ww, W[f'W_v.{l}'][s]*ww, W[f'W_o.{l}'][:,s]
a,b,c,d=head(0,2)
u,s,vt=np.linalg.svd(c,full_matrices=False); T=vt[:32].T
R=W['W_fc.0']*W['ln2_w.0']
Z=R@T
nz=np.linalg.norm(Z,axis=1); nr=np.linalg.norm(R,axis=1)
print('frac of neuron read energy in T: median %.3f'%np.median(nz**2/nr**2))
print('neuron row norms quantiles', np.quantile(nr,[0.1,0.5,0.9,0.99]).round(3))
# singular values of Z
print('Z sv', np.linalg.svd(Z,compute_uv=False).round(2))
# Also full R singular values
print('R sv', np.linalg.svd(R,compute_uv=False)[:80].round(2))
```