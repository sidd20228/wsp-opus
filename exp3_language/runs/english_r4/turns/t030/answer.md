```python
svd=np.linalg.svd
def ovl(l,j):
    g=W[f'ln1_w.{l}']; sl=slice(j*64,(j+1)*64)
    return W[f'W_o.{l}'][:,sl]@(W[f'W_v.{l}'][sl]*g)
P3=W['W_proj.3']
s=svd(P3,compute_uv=False)
print('proj3 ratios s[i]/s[i+1] around 20-35:',[(i,round(s[i]/s[i+1],2)) for i in range(18,36)])
# compare to projections into "anti" subspace: W_proj.3 columns energy fraction in top29
U,_,_=svd(P3)
fr=np.linalg.norm(U[:,:29].T@P3,axis=0)**2/np.linalg.norm(P3,axis=0)**2
print('neuron output energy frac in top29: quantiles',np.quantile(fr,[0.1,0.25,0.5,0.75,0.9]).round(2))
# W_fc.3 input side: what do MLP3 neurons read? (not logit)
# check 3.1 OV right side (input) dims count
print('3.1 OV sv',svd(ovl(3,1),compute_uv=False)[:36].round(2))
```